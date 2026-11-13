// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use((req, res, next) => {
  console.log(`${req.method} ${req.originalUrl}`);
  next();
});

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

if (!NIM_API_KEY) {
  console.warn('WARNING: NIM_API_KEY is not set');
}

const SHOW_REASONING = true;
const ENABLE_THINKING_MODE = false;

const MODEL_MAPPING = {
  'nvidia': 'nvidia/nemotron-3.5-super-vl-120b-a12b',
  'step': 'stepfun-ai/step-5-preview',
  'z.flash': 'z-ai/glm-5.3-flash',
  'kimi': 'moonshotai/kimi-k3.1-pro',
  'deepseek': 'deepseek-ai/deepseek-v4.1-flash',
  'mis': 'mistralai/mistral-large-4-1t-a52b',
  'min': 'minimaxai/minimax-m3.1-flash'
};

const resolvedCache = new Map();

const authHeaders = {
  'Authorization': `Bearer ${NIM_API_KEY}`,
  'Content-Type': 'application/json'
};

async function resolveModel(model) {
  if (MODEL_MAPPING[model]) return MODEL_MAPPING[model];
  if (resolvedCache.has(model)) return resolvedCache.get(model);

  let nimModel = null;
  try {
    const probe = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      { model, messages: [{ role: 'user', content: 'test' }], max_tokens: 1 },
      { headers: authHeaders, validateStatus: (s) => s < 500 }
    );
    if (probe.status >= 200 && probe.status < 300) nimModel = model;
  } catch (e) {}

  if (!nimModel) {
    const m = String(model || '').toLowerCase();
    if (m.includes('gpt-4') || m.includes('claude-opus') || m.includes('405b')) {
      nimModel = 'meta/llama-3.1-405b-instruct';
    } else if (m.includes('claude') || m.includes('gemini') || m.includes('70b')) {
      nimModel = 'meta/llama-3.1-70b-instruct';
    } else {
      nimModel = 'meta/llama-3.1-8b-instruct';
    }
  }

  resolvedCache.set(model, nimModel);
  return nimModel;
}

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

app.get('/v1/models', (req, res) => {
  const created = Math.floor(Date.now() / 1000);
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(id => ({
      id,
      object: 'model',
      created,
      owned_by: 'nvidia-nim-proxy'
    }))
  });
});

// TEMPORARY: lists model IDs NVIDIA actually accepts. Delete after use.
app.get('/debug/upstream-models', async (req, res) => {
  try {
    const r = await axios.get(`${NIM_API_BASE}/models`, { headers: authHeaders });
    res.json(r.data.data.map(m => m.id));
  } catch (e) {
    res.status(e.response?.status || 500).json({ error: e.message });
  }
});

app.post('/v1/chat/completions', async (req, res) => {
  const controller = new AbortController();
  res.on('close', () => controller.abort());

  try {
    const { model, messages, temperature, max_tokens, stream, ...rest } = req.body;

    if (!model || !Array.isArray(messages)) {
      return res.status(400).json({
        error: { message: '`model` and `messages` are required', type: 'invalid_request_error', code: 400 }
      });
    }

    const nimModel = await resolveModel(model);

    const nimRequest = {
      ...rest,
      model: nimModel,
      messages,
      temperature: temperature ?? 0.7,
      stream: !!stream
    };
    if (max_tokens !== undefined) nimRequest.max_tokens = max_tokens;
    if (ENABLE_THINKING_MODE) nimRequest.chat_template_kwargs = { thinking: true };

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: authHeaders,
      responseType: stream ? 'stream' : 'json',
      signal: controller.signal
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      const handleLine = (line) => {
        if (!line.startsWith('data:')) return;
        const payload = line.slice(5).trim();

        if (payload === '[DONE]') {
          res.write('data: [DONE]\n\n');
          return;
        }

        try {
          const data = JSON.parse(payload);
          const choice = data.choices?.[0];
          const delta = choice?.delta;

          if (delta) {
            const reasoning = delta.reasoning_content;
            const content = delta.content;

            if (SHOW_REASONING) {
              let out = '';
              if (reasoning) {
                if (!reasoningStarted) {
                  out += '<think>\n';
                  reasoningStarted = true;
                }
                out += reasoning;
              }
              if (content) {
                if (reasoningStarted) {
                  out += '\n</think>\n\n';
                  reasoningStarted = false;
                }
                out += content;
              }
              if (choice.finish_reason && reasoningStarted) {
                out += '\n</think>\n\n';
                reasoningStarted = false;
              }
              delta.content = out;
            } else {
              delta.content = content || '';
            }
            delete delta.reasoning_content;
          }

          res.write(`data: ${JSON.stringify(data)}\n\n`);
        } catch (e) {
          res.write(`${line}\n\n`);
        }
      };

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        lines.forEach(handleLine);
      });

      response.data.on('end', () => {
        if (buffer.trim()) handleLine(buffer);
        res.end();
      });

      response.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        res.end();
      });
    } else {
      const openaiResponse = {
        id: response.data.id || `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';

          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }

          return {
            index: choice.index,
            message: {
              role: choice.message?.role || 'assistant',
              content: fullContent,
              ...(choice.message?.tool_calls ? { tool_calls: choice.message.tool_calls } : {})
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };

      res.json(openaiResponse);
    }
  } catch (error) {
    if (axios.isCancel(error)) return; // client disconnected

    let message = error.message || 'Internal server error';
    const status = error.response?.status || 500;

    // With responseType 'stream', the error body is a stream; read it
    if (error.response?.data?.on) {
      try {
        const chunks = [];
        for await (const c of error.response.data) chunks.push(c);
        const body = JSON.parse(Buffer.concat(chunks).toString());
        message = body.error?.message || body.detail || body.message || message;
      } catch (e) {}
    } else if (error.response?.data) {
      const d = error.response.data;
      message = d.error?.message || d.detail || d.message || message;
    }

    console.error('Requested:', req.body?.model, '→ upstream URL:', error.config?.url);
    console.error('Proxy error:', status, message);

    if (!res.headersSent) {
      res.status(status).json({
        error: { message, type: 'invalid_request_error', code: status }
      });
    } else {
      res.end();
    }
  }
});

// Catch-all (works on Express 4 and 5)
app.use((req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
