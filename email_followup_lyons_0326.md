Subject: Re: Paper update — full Vysotska pipeline comparison

Hi Dr. Lyons,

Follow-up — I added Vysotska's full pipeline (sequence matcher + adaptive threshold) to the comparison, not just their thresholding. Ran everything on three datasets with DINOv2 SALAD. Here are the results:

**Closed-set (all queries have a match):**

| Dataset      | Method               | Precision | Recall | F1    |
|--------------|----------------------|-----------|--------|-------|
| GardensPoint | Ours (k=2)           | 91.0%     | 100.0% | 95.3% |
| GardensPoint | Vysotska seq.match   | 84.5%     | 100.0% | 91.6% |
| GardensPoint | Vysotska thresh only | 92.8%     | 74.6%  | 82.7% |
| Nordland-500 | Ours (k=2)           | 90.0%     | 100.0% | 94.7% |
| Nordland-500 | Vysotska seq.match   | 99.5%     | 100.0% | 99.6% |
| Nordland-500 | Vysotska thresh only | 92.5%     | 45.9%  | 61.3% |
| SFU          | Ours (k=2)           | 79.2%     | 100.0% | 88.4% |
| SFU          | Vysotska seq.match   | 99.5%     | 100.0% | 99.7% |
| SFU          | Vysotska thresh only | 83.1%     | 66.7%  | 74.0% |

**Open-set (genuine queries + distractors):**

| Dataset                  | Method               | F1    | Distractor Rej. |
|--------------------------|----------------------|-------|-----------------|
| GardensPoint (+385 SFU)  | Ours (k=2)           | 95.3% | 100%            |
| GardensPoint (+385 SFU)  | Vysotska seq.match   | 52.9% | 30%             |
| GardensPoint (+385 SFU)  | Vysotska thresh only | 49.6% | 41%             |
| Nordland-500 (+200 GP)   | Ours (k=2)           | 93.9% | 96%             |
| Nordland-500 (+200 GP)   | Vysotska thresh only | 51.4% | 30%             |

Their sequence matcher is strong in closed-set (99.6% on Nordland, 99.7% on SFU) — it was built for that scenario. But in open-set it drops to 52.9% F1 because it tries to match every query to something along the path, including distractors. It only rejects 30%.

Our method holds steady in both settings. The framing for the paper: Vysotska's approach is designed for closed-set and excels there; our contribution is extending VPR to the open-set scenario where the query stream may contain new, unseen places. We compete in closed-set and significantly outperform in open-set.

Updated the paper Results and Conclusions with this framing. Let me know your thoughts.

Nick