- Current scope (latest user direction): compare **DeepSeek-V4-Flash, GLM-5.3-Flash, and Qwen3.8-Flash-Next-FP8**. Add Kimi only if the user explicitly asks in a future session.
- Make comparison easy: use the same model order (DeepSeek → GLM → Qwen), shared architecture dimensions, consistent visual identities, and side-by-side workload metrics with exact arm qualifications.
- Keep two presentation parts: frontier architecture and efficiency mechanisms, then local measurements, interpretation and a testable research direction for SyFI.
- Investigate the architecture of the different model families and present their similarities, differences, and, most importantly, performance implications. 

Some aspects to consider,
1. For various workloads and at different batch sizes, which part of the model is the bottleneck?
2. If implementing prefix cache, what are the challenges?
3. How does speculative decoding affect the performance?
4. What is the serving cost comparison between these models?

Real measurements and careful calculations are expected. 
