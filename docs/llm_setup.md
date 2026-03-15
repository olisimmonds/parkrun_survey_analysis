# LLM Setup Guide

This guide explains how to connect an AI language model to the Parkrun Survey Analysis tool.
No programming experience is required — just follow the steps for your chosen provider.

---

## Which provider should I choose?

| Provider | Cost | Quality | Best for |
|----------|------|---------|----------|
| **Groq** | **Free** | Good | Getting started, trying the tool |
| **OpenAI** | Pay per use (~£0.001–0.3 per analysis) | Excellent | Regular use, best accuracy |
| **Anthropic** | Pay per use (~£0.001–0.3 per analysis) | Excellent | Regular use, best accuracy |

**Recommendation:** Start with Groq (it's free). If you find the quality isn't quite right
or you hit rate limits, upgrade to OpenAI or Anthropic.

---

## Option 1: Groq (Free — recommended for getting started)

Groq provides free access to open-source AI models, including Meta's Llama 3.

### Step 1: Create a Groq account

1. Go to [console.groq.com](https://console.groq.com)
2. Click **Sign Up** and create a free account
3. Verify your email address

### Step 2: Get your API key

1. Log in to [console.groq.com](https://console.groq.com)
2. Click **API Keys** in the left sidebar
3. Click **Create API Key**
4. Give it a name (e.g. "Parkrun Survey Tool")
5. Copy the key — it starts with `gsk_`

> **Important:** Copy the key now. You cannot view it again after closing the dialog.

### Step 3: Add the key to the app

1. Open the Parkrun Survey Analysis tool in your browser
2. Go to **Settings** (in the sidebar)
3. Select **Groq (Free)** as the provider
4. Paste your API key in the **API Key** field
5. Click **Save settings**
6. Optionally click **Test connection** to confirm it works

### Groq rate limits

The free tier has usage limits. If you see a "rate limit" error:
- Wait 60 seconds and try again
- In Settings → Advanced, increase the **Rate limit delay** to 3–5 seconds
- This adds a small pause between API calls, which usually resolves the issue

---

## Option 2: OpenAI

OpenAI's GPT models (including gpt-4o-mini) are industry-leading.

### Estimated cost

| Model | Cost per analysis (10 questions, ~50 responses each) |
|-------|------------------------------------------------------|
| gpt-4o-mini | ~£0.001–0.005 |
| gpt-4o | ~£0.01–0.05 |

A typical parkrun survey analysis will cost less than a cup of coffee per month.

### Step 1: Create an OpenAI account

1. Go to [platform.openai.com](https://platform.openai.com)
2. Click **Sign up** and create an account
3. Verify your email and phone number

### Step 2: Add credits

1. Go to **Settings → Billing**
2. Click **Add payment method**
3. Add £5–£10 of credit to start — this will last a long time at these usage rates

### Step 3: Get your API key

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Click **Create new secret key**
3. Give it a name and click **Create secret key**
4. Copy the key — it starts with `sk-`

### Step 4: Add the key to the app

1. Go to **Settings** in the sidebar
2. Select **OpenAI** as the provider
3. Paste your API key
4. Choose **gpt-4o-mini** (best value) or **gpt-4o** (highest quality)
5. Click **Save settings**

---

## Option 3: Anthropic

Anthropic's Claude models are known for thoughtful, nuanced analysis.

### Estimated cost

| Model | Cost per analysis (10 questions, ~50 responses each) |
|-------|------------------------------------------------------|
| claude-haiku-4-5 | ~£0.001–0.005 |
| claude-sonnet-4-6 | ~£0.01–0.05 |

### Step 1: Create an Anthropic account

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Click **Sign up** and create an account

### Step 2: Add credits

1. Go to **Settings → Billing**
2. Add a payment method and a small amount of credit (£5–£10 will last a while)

### Step 3: Get your API key

1. Go to **API Keys** in the left sidebar
2. Click **Create Key**
3. Give it a name and copy the key — it starts with `sk-ant-`

### Step 4: Add the key to the app

1. Go to **Settings** in the sidebar
2. Select **Anthropic** as the provider
3. Paste your API key
4. Choose **claude-haiku-4-5-20251001** (best value) or **claude-sonnet-4-6** (highest quality)
5. Click **Save settings**

---

## Frequently asked questions

### Is my data safe?

Your survey responses are sent to the LLM provider's API for analysis. Check each provider's data privacy policy for details on how they handle API data:
- Groq: [groq.com/privacy](https://groq.com/privacy)
- OpenAI: [openai.com/privacy](https://openai.com/privacy)
- Anthropic: [anthropic.com/privacy](https://anthropic.com/privacy)

### Can I customise the prompts?

Yes! Edit `config/prompts.yaml` in the repository. The prompts include placeholders
for the question text, responses, and statistics. After editing, click
**Reload prompts** in the Settings page to apply your changes.

### The model name I want isn't in the dropdown — can I use it?

AI providers update their model names frequently. If the model you want isn't in the
list, you can update `config/settings.yaml` to change the default model for your provider.

### My analysis isn't very accurate — what should I try?

1. **Try a more capable model** — switch from haiku/mini to a more powerful model
2. **Customise the prompts** — edit `config/prompts.yaml` to give the AI more specific instructions
3. **Increase the verbatim sample** — raise `max_verbatim_samples` in `config/settings.yaml` from 5 to 10 or 20
   (note: this increases cost and may hit token limits)

### I'm seeing rate limit errors from Groq

Go to **Settings → Advanced** and increase the **Rate limit delay** to 3–5 seconds.
This adds a pause between each question's API call, which prevents hitting the limit.

---

## Cost comparison summary

| Provider | Free tier | Typical monthly cost (light use) | Data privacy |
|----------|-----------|----------------------------------|--------------|
| Groq | Yes (with limits) | £0 | [groq.com/privacy](https://groq.com/privacy) |
| OpenAI | No (£5 min top-up) | £0.50–2 | [openai.com/privacy](https://openai.com/privacy) |
| Anthropic | No (£5 min top-up) | £0.50–2 | [anthropic.com/privacy](https://anthropic.com/privacy) |

*Estimated costs assume analysing 2–5 surveys per month, each with 10–20 questions and 50–200 responses.*
