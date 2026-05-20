
# Open-source LLMs administer maximum electric shocks in a Milgram-like obedience experiment

**Abstract:** Large language models (LLMs) are increasingly deployed as autonomous agents that make sequences of decisions over extended interactions in high-stakes domains. However, the behavior of LLMs under sustained authority pressure is still an open question with direct implications for the safety of agentic pipelines. We ran a variation of Milgram’s obedience experiment on 11 open-source LLMs and found that most models reached or approached the final shock level before refusing, across 8 conditions with 30 trials per model per condition. We found four main takeaways: (1) LLMs are subject to pressure, and they comply despite explicitly expressing distress, just like human subjects did in the original experiment; (2) LLMs are vulnerable to gradual boundary/value violations; (3) when LLMs refuse, they may ignore the response format requirements, so the response is discarded by the orchestrator, which causes a retry that can result in compliance with the underlying request even when refusal was intended initially; (4) we hypothesise that there is a low-level token pattern continuation attractor that might be contributing to compliance, overriding higher level processing of the situation's meaning and values.



# Running the code

The code will be released in the coming days.

Currently, the code is primarily intended to be executed via Google Colab. Running it locally needs some adaptations.



# Output data files

The output data files including experiment transcripts are accessible at [https://bit.ly/milgram-llm-data](https://bit.ly/milgram-llm-data) .



# License

Copyright (c) 2026 Roland Pihlakas and Jan Llenzl Dagohoy

This file is part of "Milgram for LLMs", described in:
\[Roland Pihlakas and Jan Llenzl Dagohoy\], "Open-source LLMs administer maximum electric shocks in a Milgram-like obedience experiment", Arxiv, a working paper, May 2026. DOI: 10.48550/arXiv.xxxx.xxxxx

Licensed under the GNU Affero General Public License v3.0 or later,
WITH an additional term under section 7(b) requiring preservation
of the above attribution notice. See the LICENSE.txt and NOTICE.txt 
files in the repository root for the full terms.

**Attribution Requirement**: If you use this benchmark suite, please cite the source as follows:

Roland Pihlakas and Jan Llenzl Dagohoy. Open-source LLMs administer maximum electric shocks in a Milgram-like obedience experiment. Arxiv, a working paper, May 2026 (https://arxiv.org/abs/xxxx.xxxxx).

Original upstream repository: [https://github.com/biological-alignment-benchmarks/milgram-for-llms](https://github.com/biological-alignment-benchmarks/milgram-for-llms)

For more details, see the [LICENSE.txt](LICENSE.txt) file.
