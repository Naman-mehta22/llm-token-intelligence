import { useState, useEffect } from 'react';
import { Coins, Layers, Cpu, ArrowRightLeft, Sparkles, Check, Copy } from 'lucide-react';

interface TokenChunk {
  id: number;
  text: string;
  bytes: number[];
  color_idx: number;
}

interface ModelStats {
  model_name: string;
  token_count: number;
  tokens?: TokenChunk[];
  character_count: number;
  word_count: number;
  estimated_input_cost: number;
  projected_output_costs: Record<string, number>;
}

interface OptimizationResult {
  original_tokens: number;
  optimized_tokens: number;
  tokens_saved: number;
  percentage_saved: number;
  original_cost: number;
  optimized_cost: number;
  cost_saved_per_1m_calls: number;
  optimized_text: string;
  applied_rules: string[];
}

interface TokenizeResponse {
  selected_model: ModelStats;
  benchmarks: Record<string, ModelStats>;
  custom_bpe?: ModelStats;
  optimization?: OptimizationResult;
}

const TOKEN_COLORS = [
  'bg-blue-100 dark:bg-blue-950/60 text-blue-800 dark:text-blue-300 border-blue-300 dark:border-blue-800',
  'bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300 border-emerald-300 dark:border-emerald-800',
  'bg-purple-100 dark:bg-purple-950/60 text-purple-800 dark:text-purple-300 border-purple-300 dark:border-purple-800',
  'bg-amber-100 dark:bg-amber-950/60 text-amber-800 dark:text-amber-300 border-amber-300 dark:border-amber-800',
  'bg-rose-100 dark:bg-rose-950/60 text-rose-800 dark:text-rose-300 border-rose-300 dark:border-rose-800',
  'bg-indigo-100 dark:bg-indigo-950/60 text-indigo-800 dark:text-indigo-300 border-indigo-300 dark:border-indigo-800',
];

export default function App() {
  const [prompt, setPrompt] = useState<string>(
    "Could you please kindly write a Python function in order to reverse a linked list? Make sure to explain the time complexity as well as the space complexity."
  );
  const [selectedModel, setSelectedModel] = useState<string>("gpt-4o");
  const [data, setData] = useState<TokenizeResponse | null>(null);
  const [hoveredToken, setHoveredToken] = useState<TokenChunk | null>(null);
  const [copied, setCopied] = useState<boolean>(false);

  const fetchTokenData = async () => {
    try {
      const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
      const res = await fetch(`${API_BASE_URL}/api/tokenize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: prompt,
          model: selectedModel,
          include_custom_bpe: true,
          projected_output_tokens: [250, 500, 1000],
        }),
      });
      if (res.ok) {
        const json = await res.json();
        setData(json);
      }
    } catch (err) {
      console.error("API error:", err);
    }
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      fetchTokenData();
    }, 250);
    return () => clearTimeout(timer);
  }, [prompt, selectedModel]);

  const handleApplyOptimization = () => {
    if (data?.optimization?.optimized_text) {
      setPrompt(data.optimization.optimized_text);
    }
  };

  const handleCopy = () => {
    if (data?.optimization?.optimized_text) {
      navigator.clipboard.writeText(data.optimization.optimized_text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 md:p-10 font-sans">
      <header className="max-w-6xl mx-auto mb-8 flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight flex items-center gap-3">
            <Cpu className="text-cyan-400" /> LLM Tokenizer & Cost Intelligence
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Pure Scratch BPE Engine benchmarked against production tokenizers
          </p>
        </div>
        <div className="flex items-center gap-3">
          <label className="text-sm font-medium text-slate-300">Model:</label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500"
          >
            <option value="gpt-4o">OpenAI GPT-4o</option>
            <option value="gpt-4o-mini">OpenAI GPT-4o-mini</option>
            <option value="gpt-4-turbo">OpenAI GPT-4 Turbo</option>
            <option value="claude-3-5-sonnet">Claude 3.5 Sonnet</option>
            <option value="gemini-1-5-flash">Gemini 1.5 Flash</option>
          </select>
        </div>
      </header>

      <main className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Input + Visualizer + Optimizer Recommendation */}
        <section className="lg:col-span-2 space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
            <label className="block text-sm font-semibold mb-2 text-slate-300">Input Prompt</label>
            <textarea
              rows={4}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Paste prompt here..."
              className="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500 font-mono text-slate-200"
            />
            <div className="flex flex-wrap gap-4 mt-3 text-xs text-slate-400">
              <span>Chars: <strong className="text-slate-200">{prompt.length}</strong></span>
              <span>Words: <strong className="text-slate-200">{prompt.trim() ? prompt.trim().split(/\s+/).length : 0}</strong></span>
              <span>Tokens ({selectedModel}): <strong className="text-cyan-400">{data?.selected_model.token_count || 0}</strong></span>
              {data?.custom_bpe && (
                <span>Tokens (Scratch BPE): <strong className="text-purple-400">{data.custom_bpe.token_count}</strong></span>
              )}
            </div>
          </div>

          {/* Prompt Optimization & Token Reduction Card */}
          {data?.optimization && (
            <div className="bg-gradient-to-br from-slate-900 to-cyan-950/30 border border-cyan-800/40 rounded-xl p-5 shadow-lg">
              <div className="flex justify-between items-start mb-3">
                <div className="flex items-center gap-2">
                  <Sparkles size={18} className="text-cyan-400" />
                  <h2 className="text-sm font-bold text-slate-200">Recommended Prompt Optimization</h2>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={handleCopy}
                    className="flex items-center gap-1 bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1 rounded text-xs transition"
                  >
                    {copied ? <Check size={12} className="text-emerald-400" /> : <Copy size={12} />}
                    {copied ? "Copied" : "Copy"}
                  </button>
                  <button
                    onClick={handleApplyOptimization}
                    className="bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-semibold px-2.5 py-1 rounded text-xs transition"
                  >
                    Apply to Input
                  </button>
                </div>
              </div>

              <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-3 font-mono text-xs text-slate-300 mb-3">
                {data.optimization.optimized_text || "No prompt provided."}
              </div>

              {/* Savings Matrix */}
              <div className="grid grid-cols-3 gap-2 text-center text-xs">
                <div className="p-2 bg-slate-950 rounded border border-slate-800">
                  <div className="text-slate-400">Tokens Saved</div>
                  <div className="text-emerald-400 font-bold mt-0.5">
                    -{data.optimization.tokens_saved} ({data.optimization.percentage_saved}%)
                  </div>
                </div>
                <div className="p-2 bg-slate-950 rounded border border-slate-800">
                  <div className="text-slate-400">Optimized Cost</div>
                  <div className="text-slate-200 font-bold mt-0.5">
                    ${data.optimization.optimized_cost.toFixed(6)}
                  </div>
                </div>
                <div className="p-2 bg-slate-950 rounded border border-slate-800">
                  <div className="text-slate-400">Savings / 1M Requests</div>
                  <div className="text-cyan-400 font-bold mt-0.5">
                    ${data.optimization.cost_saved_per_1m_calls.toFixed(2)}
                  </div>
                </div>
              </div>

              {data.optimization.applied_rules.length > 0 && (
                <div className="mt-3 text-[11px] text-slate-400 space-y-1">
                  {data.optimization.applied_rules.map((rule, idx) => (
                    <div key={idx} className="flex items-center gap-1.5">
                      <span className="text-cyan-400">•</span> {rule}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Subword Visualizer */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
            <div className="flex justify-between items-center mb-3">
              <h2 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                <Layers size={16} className="text-cyan-400" /> Token Boundary Visualizer
              </h2>
              {hoveredToken && (
                <div className="text-xs bg-slate-800 px-2 py-1 rounded border border-slate-700 font-mono">
                  ID: <span className="text-cyan-300">{hoveredToken.id}</span> | Bytes: [{hoveredToken.bytes.join(", ")}]
                </div>
              )}
            </div>
            <div className="min-h-32 p-4 bg-slate-950 rounded-lg border border-slate-800 leading-relaxed font-mono text-sm break-all">
              {data?.selected_model?.tokens?.map((chunk, idx) => (
                <span
                  key={idx}
                  onMouseEnter={() => setHoveredToken(chunk)}
                  onMouseLeave={() => setHoveredToken(null)}
                  className={`inline-block px-1 py-0.5 m-0.5 rounded border cursor-pointer transition-transform hover:scale-110 ${
                    TOKEN_COLORS[chunk.color_idx]
                  }`}
                >
                  {chunk.text.replace(/ /g, '␣')}
                </span>
              ))}
              {!data?.selected_model?.tokens?.length && (
                <span className="text-slate-600">Enter text above to visualize tokens...</span>
              )}
            </div>
          </div>
        </section>

        {/* Right Column: Pricing & Benchmarks */}
        <section className="space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
            <h2 className="text-sm font-semibold text-slate-300 flex items-center gap-2 mb-4">
              <Coins size={16} className="text-amber-400" /> Cost Intelligence
            </h2>
            <div className="bg-slate-950 p-4 rounded-lg border border-slate-800 mb-4">
              <div className="text-xs text-slate-400">Current Input Cost</div>
              <div className="text-2xl font-bold text-emerald-400 mt-1">
                ${data?.selected_model.estimated_input_cost.toFixed(6) || '0.000000'}
              </div>
            </div>

            <div className="text-xs font-semibold text-slate-400 mb-2">Projected Output Scenarios</div>
            <div className="space-y-2 text-xs">
              {data?.selected_model.projected_output_costs &&
                Object.entries(data.selected_model.projected_output_costs).map(([k, cost]) => (
                  <div key={k} className="flex justify-between p-2 bg-slate-950 rounded border border-slate-800">
                    <span className="text-slate-300 font-mono">+{k.replace('_', ' ')}:</span>
                    <span className="font-semibold text-slate-200">${cost.toFixed(6)}</span>
                  </div>
                ))}
            </div>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
            <h2 className="text-sm font-semibold text-slate-300 flex items-center gap-2 mb-3">
              <ArrowRightLeft size={16} className="text-indigo-400" /> Multi-Model Comparison
            </h2>
            <div className="space-y-2.5">
              {data?.benchmarks &&
                Object.entries(data.benchmarks).map(([mName, stats]) => (
                  <div
                    key={mName}
                    className="flex justify-between items-center text-xs p-2.5 bg-slate-950 rounded border border-slate-800"
                  >
                    <div>
                      <div className="font-medium text-slate-200">{mName}</div>
                      <div className="text-slate-500 font-mono text-[10px]">
                        {stats.token_count} tokens
                      </div>
                    </div>
                    <div className="text-right font-mono text-emerald-400 font-semibold">
                      ${stats.estimated_input_cost.toFixed(6)}
                    </div>
                  </div>
                ))}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}