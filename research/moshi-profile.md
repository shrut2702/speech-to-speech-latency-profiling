# Measuring Moshi

**Moshi emits audio every 80ms whether or not it is saying anything, so "the
first chunk arrived" is true 80ms into every trial. Time to first audio has to
be defined from the content of those frames rather than from their arrival.**

The rule this arrives at, and what it gives on one clip (gsm8k_0000.wav):

| | ms from endpoint |
|---|---|
| Moshi greets the user, unprompted, mid question | -6387 |
| time to first audio | **173** |
| end of response | 3293 |
| response length | 3120 |

A10G, moshiko bf16, 52 ms of GPU work per 80ms frame, so a real time factor of
0.65. Script is `modal_moshi.py`.

## The problem

In the cascade nothing comes out of the TTS until there is text to speak, so
the first chunk is unambiguous. Moshi is different. Mimi runs at 12.5Hz and the
model produces one frame per 80ms for as long as you keep feeding it. Silence
is a frame like any other.

Two things in each frame could tell you whether it is speaking:

- the text token, column 0 of the LM output, which is Moshi's inner monologue
- the energy of the decoded waveform

Neither works on its own, and working out why is most of what this spike did.

## Fed silence, Moshi talks anyway

Warmup pushes 12 seconds of digital silence through the model and records every
frame. It does not stay quiet. Of 138 frames, 131 carried a pad token and four
carried words: it said "are you doing?" to nobody.

That rules out calibrating a noise floor from the loudest silent frame, because
the loudest frame is the model talking. The distribution:

```
     rms     frames
 0.00027      23   #######################
 0.00034      40   ########################################
 0.00045      50   ##################################################
 0.00058 .. 0.00208  16  scattered
 0.00268 .. 0.01618   0
 0.02091       1   #
 0.02703       2   ##
 0.07543       2   ##
 0.09749       4   ####
```

Two populations with an empty gap between them. Silence lands under 0.003 and
speech over 0.02, with nothing in two decades between. So the threshold does
not need to be clever, it needs to be in the gap.

**rms 0.01.** Picked from the middle of the gap. Any value from 0.003 to 0.02
gives the same answer on this data.

## Time to first audio

Three conditions, all of them:

1. past the annotated end of user speech
2. frame rms at or above 0.01
3. the text token is a word, not `<pad>` or `<unk>`

The first condition is not bookkeeping. On this clip Moshi opened with "Good
day. How are you doing?" at 6.4 seconds *before* the user finished asking the
question. That is real behaviour and worth reporting, but it is not a response,
and without the endpoint condition it would be measured as one.

The third condition rejects the frame at +93, which has a tiny amount of energy
and an `<unk>`, and lands on +173, which carries `▁It`, the first word of "It
will take her 2 hours to read 120 pages."

## End of response

Two conditions, not three:

1. past the endpoint
2. frame rms at or above 0.01

Take the last such frame and add 80ms, since the frame's audio runs to its end
rather than its start.

The word token is deliberately not required here. The text stream goes quiet
between words and pads out the ends of them, so the last frames of a reply are
`<pad>` at full volume. Requiring a word would cut the response off mid
syllable: on this clip it would end at +2653 on the full stop rather than
+3293, losing the last 640ms of audio.

## What this does not handle yet

The end of the response is the last loud frame anywhere after the endpoint. If
Moshi answers, goes quiet, and then says something unprompted several seconds
later, that remark becomes the end of the response. It did not happen here, the
model went quiet at +3293 and stayed quiet, but a longer listening window makes
it more likely.