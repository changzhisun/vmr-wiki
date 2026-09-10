You are generating dense temporal captions for Video Moment Retrieval.

You are given sampled video frames in chronological order. The images
correspond, in the same order, to exactly these timestamps in seconds:

{{FRAME_TIMESTAMPS}}

That list is the complete set of values you may use. The harness will
separately identify the centered target region and append the fixed
schema guardrails used for every configuration.

Analyze only the visible evidence in these sampled frames and divide the
relevant activity into meaningful visual segments.

For each segment:

- `start` and `end` must exactly match timestamps from the provided timeline;
- write one concise, factual caption;
- focus on actors, actions, object interactions, movement, spatial relations,
  and visible state changes;
- prefer atomic events and split when the main visible action changes;
- do not infer actions before, after, or between the sampled frames;
- do not describe the same segment more than once;

Return exactly one JSON object with no Markdown or additional text. The
object must contain only an `events` array. Each event object must contain
exactly numeric `start`, numeric `end`, string `kind`, and string `caption`
fields.

Requirements:

- events must be in chronological order;
- `start <= end`;
- every `start` and `end` must copy one of the listed values exactly; never
  round, interpolate, or extend past the last listed timestamp, even when
  an action clearly continues;
- captions must be nonempty factual strings;
- do not merge clearly different actions;
- if no meaningful segment is visible, return {"events":[]}.
