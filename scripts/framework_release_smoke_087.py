"""Actual installed-framework HTTP, WebSocket and package-only demo speech."""
import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import wave

import numpy as np
from fastapi.testclient import TestClient
from huggingface_hub import hf_hub_download
import RealtimeTTS
from RealtimeTTS.qwen_server import QwenHttpServer, create_app

parser = argparse.ArgumentParser()
parser.add_argument("--gpu",action="store_true")
parser.add_argument("--cache-dir",default="model-cache")
parser.add_argument("--sample",type=Path,default=Path("samples/pr-wheel-audio/cpu_greedy_20f_customvoice_bf16.wav"))
args = parser.parse_args()
device = "gpu" if args.gpu else "cpu"
assert importlib.metadata.version("realtimetts") == "0.8.8"
assert Path(RealtimeTTS.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
if args.gpu:
    import qwentts_cpp as native
    from RealtimeTTS.engines.qwen_engine import QwenEngine as Engine
    expected_native = "0.2.0"
    extra = {}
    os.environ.pop("QWENTTS_CPU_STARTUP_PRIORITY",None)
else:
    import qwentts_cpp_cpu as native
    from RealtimeTTS.engines.qwen_cpu_engine import QwenCpuEngine as Engine
    expected_native = "0.4.0"
    extra = dict(cpu_threads=2,cpu_codec_threads=2,cpu_stream_frames=2,
                 onset_silence_recovery=True)
    os.environ["QWENTTS_CPU_STARTUP_PRIORITY"] = "second_chunk"
assert native.__version__ == expected_native
assert Path(native.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
if args.gpu:
    assert any((Path(native.__file__).resolve().parent / "lib").glob("*ggml-cuda*"))
else:
    assert native.QwenLibrary().cpu_only()
files = {
    "qwen-talker-0.6b-base-Q8_0.gguf":"d54dbaf10591421fa764ed630d764efa717ae40cd959bd48c66d4eb1af226426",
    "qwen-tokenizer-12hz-Q8_0.gguf":"1883beeed99348fc35e23dd225e9082f93f6f8c109330a33d935baa8acdbfd94",
}
models = []
for name, expected in files.items():
    path = Path(hf_hub_download(repo_id="Serveurperso/Qwen3-TTS-GGUF",
        revision="e0f336a048a3de02b29b8ad92969217d9ecffe3e",filename=name,cache_dir=args.cache_dir))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    models.append(path)
engine = Engine(model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",talker_path=models[0],
    codec_path=models[1],use_fa=False,voice_cache_dir=Path("voice-cache")/device,
    warmup=False,max_new_tokens=32,do_sample=False,seed=42,clone_mode="speaker_only",
    onset_silence_profile="qwen3_tts_12hz_0_6b_base_q8_v1",startup_buffer_ms=80,**extra)
server = QwenHttpServer(engine,voice_dir=Path("registered-voices")/device,language="english",
                       startup_warmup_routes=False,api_key="local-release-rehearsal")
headers = {"Authorization":"Bearer local-release-rehearsal"}
wave_data = None
ws_chunks = []
with TestClient(create_app(server)) as client:
    assert client.get("/health").status_code == 200
    assert client.get("/v1/capabilities",headers=headers).status_code == 200
    registered = client.post("/v1/audio/voices",headers=headers,json={
        "name":"release-smoke","wav_b64":base64.b64encode(args.sample.read_bytes()).decode("ascii")})
    assert registered.status_code in (200,201), registered.text
    options = {"voice":"release-smoke","language":"english","max_new_tokens":32,
               "do_sample":False,"seed":42}
    response = client.post("/v1/audio/speech",headers=headers,
        json={**options,"input":"Release smoke.","response_format":"wav"})
    assert response.status_code == 200, response.text[:500]
    wave_data = response.content
    with wave.open(io.BytesIO(wave_data),"rb") as wav:
        assert (wav.getframerate(),wav.getsampwidth(),wav.getnchannels()) == (24000,2,1)
        pcm = np.frombuffer(wav.readframes(wav.getnframes()),dtype="<i2")
        assert pcm.size and np.max(np.abs(pcm.astype(np.int32))) > 0
    with client.websocket_connect("/v1/audio/speech-stream",headers=headers) as ws:
        ws.send_json({"type":"config",**options,"response_format":"pcm"})
        ws.send_json({"type":"text","text":"Release smoke."})
        ws.send_json({"type":"flush"})
        ws.send_json({"type":"end"})
        for _ in range(1000):
            message = ws.receive()
            if message.get("bytes") is not None:
                ws_chunks.append(message["bytes"])
            elif message.get("text"):
                event = json.loads(message["text"])
                assert event.get("type") != "error", event
                if event.get("type") == "done":
                    break
            else:
                raise AssertionError(message)
        else:
            raise AssertionError("No WebSocket done event")
ws_pcm = b"".join(ws_chunks)
assert len(ws_pcm) > 0
assert np.max(np.abs(np.frombuffer(ws_pcm,dtype="<i2").astype(np.int32))) > 0
Path(f"framework-{device}-http.wav").write_bytes(wave_data)
with wave.open(f"framework-{device}-ws.wav","wb") as output:
    output.setparams((1,2,24000,0,"NONE","not compressed"))
    output.writeframes(ws_pcm)
demo = [sys.executable,"-I","-m","RealtimeTTS.qwen_emotions","--device",device,
        "--no-play","--emotions","neutral","--text","Release smoke.","--color","never",
        "--model",str(models[0]),"--codec",str(models[1]),"--cache-dir","demo-cache",
        "--output-dir",f"demo-{device}"]
if not args.gpu:
    demo += ["--cpu-threads","2","--cpu-codec-threads","2","--cpu-stream-frames","2"]
if os.environ.get("QWEN_TRACE_SHUTDOWN") == "1":
    demo[2:4] = ["-c", "import faulthandler; faulthandler.dump_traceback_later(25); from RealtimeTTS.qwen_emotions import main; raise SystemExit(main())"]
subprocess.run(demo,check=True,timeout=60 if os.environ.get("QWEN_TRACE_SHUTDOWN") == "1" else 300)
waves = list(Path(f"demo-{device}").glob("*.wav"))
assert waves and all(p.stat().st_size > 44 for p in waves)
result = {"framework":"0.8.8","native":expected_native,"device":device,
          "module":RealtimeTTS.__file__,"http_bytes":len(wave_data),"ws_pcm_bytes":len(ws_pcm),
          "demo_wavs":[p.name for p in waves],"installed_imports":True,"passed":True}
Path(f"framework-{device}-smoke.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result))
