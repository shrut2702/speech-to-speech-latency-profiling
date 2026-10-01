
## Pipeline, from end of speech

| config | n | time to first audio p50/p95 | end to end p50/p95 | asr final p50/p95 | llm first token p50/p95 | tts first chunk p50/p95 |
|---|---|---|---|---|---|---|
| moshi | 160 | 145 / 319 | 3413 / 11947 | - | - | - |

## Time to first audio by length bucket

| config | short p95 | medium p95 | long p95 |
|---|---|---|---|
| moshi | 214 | 174 | 553 |

## Streaming health

| config | rtf p95 | max gap p95 ms | underruns | failed trials |
|---|---|---|---|---|
| moshi | 1.00 | 4 | 0 | 20 |
