from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-rw-1b")
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-rw-1b")

# ensure padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device="cpu"
)

def generate_text(summary, prompt):
    input_text = f"{summary}\n\nQ: {prompt}\nA:"
    out = generator(
      input_text,
      max_new_tokens=120,
      do_sample=True,
      top_p=0.9,
      temperature=0.7,
      repetition_penalty=1.2,
      no_repeat_ngram_size=3,
      pad_token_id=tokenizer.pad_token_id,
    )
    return out[0]["generated_text"][len(input_text):].strip()
