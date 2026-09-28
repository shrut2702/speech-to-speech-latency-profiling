!! response length varied, latencies are not comparable:
   gsm8k_0048: response length varied across trials/configs [54, 55]
   gsm8k_0220: response length varied across trials/configs [59, 60]
   gsm8k_0315: response length varied across trials/configs [87, 95]
   gsm8k_0432: response length varied across trials/configs [36, 37]
   gsm8k_0472: response length varied across trials/configs [63, 64]
   llamaq_0152: response length varied across trials/configs [28, 29]
   llamaq_0166: response length varied across trials/configs [25, 27, 29, 31]
   llamaq_0246: response length varied across trials/configs [25, 31]
   mlcpro_0026: response length varied across trials/configs [48, 55]
   mlcpro_0074: response length varied across trials/configs [58, 65]


## Pipeline, from end of speech

| config | n | time to first audio p50/p95 | end to end p50/p95 | asr final p50/p95 | llm first token p50/p95 | tts first chunk p50/p95 |
|---|---|---|---|---|---|---|
| cascade_stream_all_cosyvoice2 | 180 | 2947 / 4298 | 13592 / 22700 | 4 / 672 | 31 / 702 | 2947 / 4298 |

## Per stage, from each stage's own start

Stage costs with everything before them subtracted out. This is the view that says which millisecond to go delete.

| config | n | asr first partial p50/p95 | asr total p50/p95 | llm ttft p50/p95 | llm 2nd token p50/p95 | llm total p50/p95 | tts first audio p50/p95 | tts total p50/p95 |
|---|---|---|---|---|---|---|---|---|
| cascade_stream_all_cosyvoice2 | 180 | 2123 / 4122 | 4112 / 9384 | 26 / 34 | 7 / 8 | 431 / 717 | 2802 / 4056 | 13471 / 22550 |

## Time to first audio by length bucket

| config | short p95 | medium p95 | long p95 |
|---|---|---|---|
| cascade_stream_all_cosyvoice2 | 4191 | 3899 | 4704 |

## Streaming health

| config | rtf p95 | max gap p95 ms | underruns | failed trials |
|---|---|---|---|---|
| cascade_stream_all_cosyvoice2 | 1.11 | 3273 | 195 | 0 |
