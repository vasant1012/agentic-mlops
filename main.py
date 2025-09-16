"""
MLflow -> LangChain Agent prototype
Tools:
 - get_runs_summary: query MLflow experiment runs (params, metrics, status)
 - summarize_runs: LLM-powered summary of runs
 - recommend_hparams: LLM-powered hyperparameter suggestions (next trials)

How to adapt to Qwen:
 - Implement the `LLMAdapter` class methods to call your Qwen/Qwen2-1.5B
    endpoint or a local HF pipeline.
"""
# from local_hugging_face import LocalHFAdapter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from agent_helper import build_agent
from logger import logger


# pipe = LocalHFAdapter()
# llm = HuggingFacePipeline(pipeline=pipe)
# Load Qwen2 model + tokenizer (CPU only, fp16 disabled)
model_id = "Qwen/Qwen2-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")

# HuggingFace pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.2,
)

# Wrap in LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)
agent = build_agent(llm)
out = agent.invoke(
    {"input": f"summarize_runs experiment_id={0}"}, pydev_do_not_trace = True)  # NOQA E501
logger.info(out)


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--experiment-id", required=True,
#                         help="MLflow experiment id")
#     parser.add_argument("--action", default="summarize",
#                         choices=["summarize", "recommend", "raw"])
#     args = parser.parse_args()

#     # small CPU-friendly model
#     try:
#         llm = LocalHFAdapter()
#         agent = build_agent(llm)
#         logger.info('Agent is loaded!')
#         if args.action == "summarize":
#             out = agent.invoke(
#                 {"input": f"summarize_runs experiment_id={args.experiment_id}"}) # NOQA E501
#             logger.info(out)
#         elif args.action == "recommend":
#             out = agent.invoke({"input":
#                                         f"recommend_hparams experiment_id={args.experiment_id}"})  # NOQA E501
#             logger.info(out)
#         elif args.action == "raw":
#             out = agent.invoke({"input":
#                                         f"get_runs_summary experiment_id={args.experiment_id}"})  # NOQA E501
#             logger.info(out)
#     except:  # NOQA E722
#         logger.error('Model is not loaded.')
