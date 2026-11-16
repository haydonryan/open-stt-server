"""Raw-PCM ingress for the official NeMo cache-aware RNNT pipeline.

Only received PCM enters this adapter. No file, duration, or reference is accepted.
NeMo's bounded raw/feature rings and encoder/decoder caches persist between calls.
"""

import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.inference.factory.cache_aware_pipeline_builder import CacheAwarePipelineBuilder
from nemo.collections.asr.inference.streaming.framing.request import Frame
from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions


DEFAULT_CHECKPOINT = '/models/nemotron-speech-streaming-en-0.6b.nemo'
CHECKPOINT = os.environ.get('NEMOTRON_CHECKPOINT', DEFAULT_CHECKPOINT)
REVISION = 'ebe59e5a817142986528bbbee5dba8db7b38ed50'
CONFIG_PATH = Path(__file__).with_name('cache-aware-rnnt.yaml')


class NemotronAdapter:
    def __init__(self, device='cpu', threads=8, checkpoint=None):
        if device not in ('cpu', 'cuda'):
            raise ValueError('device must be cpu or cuda')
        checkpoint = checkpoint or CHECKPOINT
        torch.set_num_threads(threads)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; CPU fallback is forbidden')
        cfg = OmegaConf.load(CONFIG_PATH)
        cfg.asr.model_name = checkpoint
        cfg.asr.device = device
        cfg.asr.compute_dtype = 'bfloat16' if device == 'cuda' else 'float32'
        cfg.asr.use_amp = False
        cfg.asr.decoding.strategy = 'greedy_batch'
        cfg.asr.decoding.greedy.use_cuda_graph_decoder = False
        cfg.asr.decoding.greedy.enable_per_stream_biasing = False
        cfg.streaming.batch_size = 1
        cfg.streaming.num_slots = 1
        cfg.streaming.att_context_size = [70, 6]
        cfg.streaming.use_cache = True
        cfg.streaming.use_feat_cache = True
        cfg.streaming.request_type = 'frame'
        cfg.streaming.chunk_size_in_secs = None
        cfg.endpointing.stop_history_eou = -1  # Explicit client EOF, as in the Voxtral test.
        cfg.endpointing.residue_tokens_at_end = 0
        cfg.return_tail_result = False
        cfg.enable_itn = False
        cfg.enable_nmt = False
        self.pipeline = CacheAwarePipelineBuilder.build(cfg)
        self.device = device
        actual_dtype = next(self.pipeline.asr_model.asr_model.parameters()).dtype
        expected_dtype = torch.bfloat16 if device == 'cuda' else torch.float32
        if actual_dtype != expected_dtype:
            raise RuntimeError(f'unexpected weights dtype: {actual_dtype}, expected {expected_dtype}')
        self.frame_samples = round(self.pipeline.chunk_size_in_secs * 16000)
        self.metadata = {
            'model': 'nvidia/nemotron-speech-streaming-en-0.6b',
            'revision': REVISION,
            'runtime': 'NeMo 3.0.0 CacheAwareRNNTPipeline / raw Frame requests',
            'device': device,
            'weights_dtype': str(actual_dtype).removeprefix('torch.'),
            'threads': torch.get_num_threads(),
            'attention_context': [70, 6],
            'input_frame_ms': 80,
            'native_frame_samples': self.frame_samples,
            'native_raw_ring_samples': self.pipeline.bufferer.audio_bufferers[0].buffer_size,
            'native_feature_ring_frames': self.pipeline.bufferer.feature_buffer_len,
            'pending_pcm_bound_samples': self.frame_samples + 1280,
            'eof_policy': 'Hold one full native frame until next ingress or EOF; pad final frame only, with valid length.',
            'extra_ingress_hold_ms': 80,
            'endpointing': 'Explicit EOF; native automatic EOU detection disabled',
            'future_audio_access': False,
            'encoder_cache': True,
            'frontend_feature_cache': True,
            'checkpoint': checkpoint,
            'checkpoint_load_path': 'Official NeMo wrapper restores checkpoint to requested device before precision cast',
        }
        if device == 'cuda':
            self.metadata.update(torch_loaded_allocated_mib=torch.cuda.memory_allocated()/2**20,
                                 torch_loaded_reserved_mib=torch.cuda.memory_reserved()/2**20,
                                 torch_load_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        self.reset()

    def reset(self):
        # Complete requests free their slots natively. Also handle client aborts.
        if self.pipeline.context_manager.streamidx2slotidx:
            self.pipeline.context_manager.reset()
        bufferer = self.pipeline.bufferer
        occupied = list(bufferer.slotidx2streamidx)
        if occupied:
            bufferer.reset_slots(occupied)
            bufferer.free_slots(occupied)
        self.pipeline.reset_session()
        self.pending = np.empty(0, dtype=np.float32)
        self.started = False
        self.finished = False
        self.final_parts = []
        self.text = ''
        self.stats = {'received_samples': 0, 'feed_calls': 0, 'native_steps': 0,
                      'max_pending_samples': 0, 'native_eof_calls': 0,
                      'processed_real_samples': 0, 'native_final_response_is_last': False}

    def _step(self, samples, final):
        valid = len(samples)
        if not (0 < valid <= self.frame_samples):
            raise ValueError('native frame must contain received audio')
        pcm = torch.zeros(self.frame_samples, dtype=torch.float32)
        pcm[:valid] = torch.from_numpy(np.array(samples, dtype=np.float32, copy=True))
        request = Frame(samples=pcm, stream_id=0, is_first=not self.started,
                        is_last=final, length=valid,
                        options=ASRRequestOptions(stop_history_eou=-1) if not self.started else None)
        with torch.inference_mode():
            response, = self.pipeline.transcribe_step([request])
        self.started = True
        self.stats['native_steps'] += 1
        self.stats['processed_real_samples'] += valid
        if response.final_transcript:
            self.final_parts.append(response.final_transcript.strip())
        if final:
            self.stats['native_eof_calls'] += 1
            self.stats.update(native_final_response_is_last=request.is_last,
                              native_final_field_present=hasattr(response, 'final_transcript'),
                              native_state_freed=not bool(self.pipeline._state_pool),
                              native_feature_slot_freed=not bool(self.pipeline.bufferer.streamidx2slotidx),
                              native_encoder_slot_freed=not bool(self.pipeline.context_manager.streamidx2slotidx))
            if not all(self.stats[key] for key in ['native_final_field_present', 'native_state_freed',
                                                   'native_feature_slot_freed', 'native_encoder_slot_freed']):
                raise RuntimeError('native EOF did not finalize and free streaming state')
            if self.device == 'cuda':
                self.stats.update(torch_allocated_mib=torch.cuda.memory_allocated()/2**20,
                                  torch_reserved_mib=torch.cuda.memory_reserved()/2**20,
                                  torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
            # Never replace missing/empty native final text with partial text.
            self.text = ' '.join(self.final_parts).strip()
        else:
            self.text = ' '.join(self.final_parts + [response.partial_transcript.strip()]).strip()
        return self.text

    def feed(self, pcm_float32):
        if self.finished:
            raise RuntimeError('audio after EOF; call reset before a new stream')
        incoming = np.asarray(pcm_float32, dtype=np.float32)
        if incoming.ndim != 1 or not 0 < incoming.size <= 1280:
            raise ValueError('feed requires 1..1280 fresh mono samples at 16 kHz')
        if not np.isfinite(incoming).all():
            raise ValueError('non-finite PCM')
        self.stats['received_samples'] += incoming.size
        self.stats['feed_calls'] += 1
        self.pending = np.concatenate((self.pending, incoming))
        self.stats['max_pending_samples'] = max(self.stats['max_pending_samples'], self.pending.size)
        answer = None
        # Strict > deliberately retains a full final frame until EOF is known.
        while self.pending.size > self.frame_samples:
            answer = self._step(self.pending[:self.frame_samples], final=False)
            self.pending = self.pending[self.frame_samples:].copy()
        return answer

    def finish(self):
        if self.finished:
            raise RuntimeError('duplicate EOF')
        self.finished = True
        if self.pending.size:
            self._step(self.pending, final=True)
            self.pending = np.empty(0, dtype=np.float32)
        else:
            self.text = ''
        if self.stats['processed_real_samples'] != self.stats['received_samples']:
            raise AssertionError('received PCM was not processed exactly once')
        return self.text

    def close(self):
        self.reset()


def make_adapter(model='nemotron', device='cpu', threads=8, checkpoint=None):
    if model != 'nemotron':
        raise ValueError('this adapter only supports nemotron')
    return NemotronAdapter(device=device, threads=threads, checkpoint=checkpoint)
