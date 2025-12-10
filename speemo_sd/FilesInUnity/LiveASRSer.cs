using UnityEngine;
using System;
using SocketIOClient; // if you use the doghappy client (or wrapper)
using System.Threading.Tasks;

public class LiveASRSer : MonoBehaviour {
    [SerializeField] string serverUrl = "http://127.0.0.1:5000";
    [SerializeField] string modelChoice = "your_checkpoint_base";

    SocketIO socket;
    int inputSR;
    float[] accum = Array.Empty<float>();

    async void Start() {
        inputSR = AudioSettings.outputSampleRate; // often 48000
        await SetupSocket();
        SetupMicPlayback();
    }

    async Task SetupSocket(){
        socket = new SocketIO(serverUrl);
        socket.On("asr_emotion_result", resp => {
            Debug.Log(resp.GetValue().ToString());
        });
        await socket.ConnectAsync();
    }

    void SetupMicPlayback(){
        var clip = Microphone.Start(null, true, 5, inputSR);
        var src = gameObject.AddComponent<AudioSource>();
        src.loop = true; src.clip = clip; src.Play();
    }

    void OnAudioFilterRead(float[] data, int channels){
        // downmix to mono
        int frames = data.Length / channels;
        var mono = new float[frames];
        for (int i=0;i<frames;i++){
            float s=0f; for (int c=0;c<channels;c++) s += data[i*channels+c];
            mono[i] = s / channels;
        }
        // accumulate ~500ms
        int need = (int)(0.5f * inputSR);
        accum = Concat(accum, mono);
        if (accum.Length >= need){
            var res = ResampleLinear(accum, inputSR, 16000);
            byte[] payload = new byte[res.Length * sizeof(float)];
            Buffer.BlockCopy(res, 0, payload, 0, payload.Length);
            var meta = new { format="f32le", sr=16000, ch=1, ts=DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() };
            socket.EmitAsync("unity_audio_chunk", new { model_choice=modelChoice, meta, audio=payload });
            accum = Array.Empty<float>();
        }
    }

    static float[] Concat(float[] a, float[] b){
        var r = new float[a.Length+b.Length];
        Buffer.BlockCopy(a,0,r,0,a.Length*sizeof(float));
        Buffer.BlockCopy(b,0,r,a.Length*sizeof(float),b.Length*sizeof(float));
        return r;
    }
    static float[] ResampleLinear(float[] x, int sr0, int sr1){
        if (sr0==sr1) return x;
        int N = (int)Math.Floor(x.Length * (double)sr1 / sr0);
        var y = new float[N]; double r = (double)sr1/sr0;
        for (int i=0;i<N;i++){ double t=i/r; int k=(int)Math.Floor(t); double f=t-k;
            float a = x[Math.Min(k, x.Length-1)], b = x[Math.Min(k+1, x.Length-1)];
            y[i] = (float)((1.0-f)*a + f*b);
        }
        return y;
    }
}
