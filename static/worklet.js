class MicCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const ch0 = inputs[0][0];
    if (ch0) this.port.postMessage(ch0);
    return true;
  }
}
registerProcessor('mic-capture', MicCaptureProcessor);
