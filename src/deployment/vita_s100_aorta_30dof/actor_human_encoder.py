"""Pinned human encoder (including FSQ) + original shared action decoder.

Both networks run on the Python side through ORT or the existing SDK inference
session. The quantized encoded_tokens are used, never the raw_latent output.
"""
import importlib
import platform
import time

import numpy as np

from .actor_smpl import parse_contract


class Network:
    def __init__(self,path,math_api,backend,threads,*,input_name,input_dim,output_name,output_dim):
        if type(threads) is not int or threads < 1:
            raise ValueError('Inference thread count must be positive')
        self.backend=backend;self.input=input_name;self.output=output_name;self.input_dim=input_dim;self.output_dim=output_dim
        if backend=='ort':
            import onnxruntime as ort
            options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
            self.session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
            inputs=self.session.get_inputs();outputs=self.session.get_outputs();self.metadata=self.session.get_modelmeta().custom_metadata_map
            name=lambda x:x.name;shape=lambda x:x.shape;dtype=lambda x:x.type
        elif backend=='sdk':
            session_type=importlib.import_module(math_api.__name__.rpartition('.')[0]+'._onnx').Session
            self.session=session_type(str(path),intra_threads=threads,inter_threads=1)
            inputs=self.session.inputs;outputs=self.session.outputs;self.metadata=self.session.metadata
            name=lambda x:x.name;shape=lambda x:x.shape;dtype=lambda x:x.dtype
        else:
            raise ValueError('Unknown human encoder inference backend')
        selected=[o for o in outputs if name(o)==output_name]
        if (len(inputs)!=1 or name(inputs[0])!=input_name or shape(inputs[0])!=[1,input_dim]
                or dtype(inputs[0])!='tensor(float)' or len(selected)!=1
                or shape(selected[0])!=[1,output_dim] or dtype(selected[0])!='tensor(float)'):
            raise ValueError('Pinned human encoder/decoder tensor contract mismatch')

    def run(self,x):
        x=np.asarray(x,dtype=np.float32)
        if x.shape!=(1,self.input_dim) or not np.isfinite(x).all():
            raise ValueError('Invalid human encoder/decoder input')
        x=np.ascontiguousarray(x)
        out=(self.session.run([self.output],{self.input:x})[0] if self.backend=='ort'
             else self.session.run(self.input,self.output,x))
        if out.shape!=(1,self.output_dim) or not np.isfinite(out).all():
            raise ValueError('Invalid human encoder/decoder output')
        return out


class HumanEncoderDecoder:
    def __init__(self,encoder,decoder,math_api,sidecar,*,backend='auto',threads=1):
        if backend=='auto':backend='sdk' if platform.machine()=='aarch64' else 'ort'
        self.backend=backend
        self.contract=parse_contract(math_api,sidecar,{})
        self.encoder=Network(encoder,math_api,backend,threads,input_name='human_features',input_dim=840,output_name='encoded_tokens',output_dim=64)
        self.decoder=Network(decoder,math_api,backend,threads,input_name='obs_dict',input_dim=964,output_name='action',output_dim=28)
        if (self.encoder.metadata.get('checkpoint_iteration')!='27000'
                or self.encoder.metadata.get('source_onnx_sha256')!=sidecar['model']['sha256']):
            raise ValueError('Human encoder and source SMPL actor provenance differ')
        self.last_tokens=None;self.last_timing=None

    def infer(self,observation):
        obs=np.asarray(observation,dtype=np.float32)
        if obs.shape!=(1740,) or not np.isfinite(obs).all():
            raise ValueError('Expected finite SMPL observation [1740]')
        start=time.perf_counter()
        tokens=self.encoder.run(obs[:840][None])
        middle=time.perf_counter()
        decoder_input=np.concatenate([tokens,obs[840:][None]],axis=1)
        result=self.decoder.run(decoder_input)
        end=time.perf_counter();self.last_tokens=tokens[0].copy()
        self.last_timing=dict(human_encoder_ms=(middle-start)*1000,action_decoder_ms=(end-middle)*1000,total_ms=(end-start)*1000)
        return result[0]
