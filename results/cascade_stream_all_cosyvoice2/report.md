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
