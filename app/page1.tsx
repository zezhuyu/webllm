"use client";
import { useEffect, useState, useRef, SetStateAction } from "react";
import { AutoTokenizer, pipeline, env, TextGenerationPipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers/dist/transformers.min.js';
// import { pipeline, AutoTokenizer } from "@huggingface/transformers";

export default function Home() {

  const [tokens, setTokens] = useState<string[]>([]);
  const [token, setToken] = useState<string>("");
  // const [tokenizer, setTokenizer] = useState<AutoTokenizer>("");
  // const [pipe, setPipe] = useState<TextGenerationPipeline>("");
  const [progress, setProgress] = useState<number>(100);
  const [isready, setReady] = useState<boolean>(false);
  const [question, setQuestion] = useState<string>("");
  const [e, setE] = useState<string>("");
  const [device, setDevice] = useState<string>("cpu");
  const [memory, setMemory] = useState<string>("0");
  const hasInitialized = useRef(false);
  // const tokenizer = useRef<AutoTokenizer>();
  const pipe = useRef<TextGenerationPipeline>();

  const pipeConfig = {
    dtype: 'q4f16',
    progress_callback: (p: { status: string; progress: SetStateAction<number>; }) => {
      if (p.status === "initiate"){
        setProgress(0);
        setReady(false);
      }else if (p.status === "progress"){
        setProgress(Math.floor(Number(p.progress)));
        // setReady(false);
      }else if (p.status === "ready"){
        setProgress(100);
        setReady(true);
      }
    }
  }
  useEffect(() => {
    if (hasInitialized.current) {
      return;
    }
    hasInitialized.current = true;

    if((navigator as any).deviceMemory){
      setMemory(navigator .deviceMemory);
    }

    if((navigator as any).gpu){
      pipeConfig.device = 'webgpu';
      setDevice('webgpu');
    }
    
    async function init(){
      try {
        env.localModelPath = './models';
        // env.backends.onnx.wasm.wasmPaths = './wasm/ort-wasm-simd-threaded.jsep.wasm';
        env.allowLocalModels = true;
        env.allowRemoteModels = false; 
        // tokenizer.current = await AutoTokenizer.from_pretrained("Llama-3.2-1B-Instruct");
        pipe.current = await pipeline("text-generation", "Llama-3.2-1B-Instruct", pipeConfig);
        // console.log(pipe.current.tokenizer);
        // setTokenizer(tokeni);
        // setPipe(pipel);
        return;
      } catch (error) {
        console.log(error);
        setE(error.message);
      }
      
    };
    init();
  }, []);

  class addToken {

    constructor() {
      setTokens([]);
      setToken("");
    }
    
    put(list: string[]) {
      setTokens(pipe.current.tokenizer.decode(list[0], {skip_special_tokens: true,}));
      setToken((prevToken) => prevToken + (pipe.current ? pipe.current.tokenizer.decode(list[0], { skip_special_tokens: true }) : ''));
      // setToken(token + pipe.current.tokenizer.decode(list[0], {skip_special_tokens: true,}));
      // console.log(tokenizer.decode(list[0], {skip_special_tokens: true,}))
    }
    
    end() {
      // setTokens([]);
      // setToken(""); 
    }
  }

  const handleSubmit = async (event: {
    target: { question: any; }; preventDefault: () => void; 
}) => {
    try {
      event.preventDefault();
      if (pipe.current !== null && pipe.current !== "" && typeof pipe.current === "function" && question !== "") {
        // console.log(pipe.current.tokenizer);
        const output = await pipe.current(question, { 
          max_new_tokens: 1024,
          streamer: new addToken(),
       })
       console.log(output[0].generated_text);
      }
      // setPipe(pipel);
      return ;
    } catch (error) {
      console.log( error);
      setE(error.message);
    }
    
  };

  return (
    <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      <main className="w-full mx-auto flex flex-col gap-8 row-start-2 items-center sm:items-start">
        <form onSubmit={handleSubmit} className="w-full max-w-xl mx-auto">   
            <label htmlFor="default-search" className="mb-2 text-sm font-medium text-gray-900 sr-only dark:text-white">Ask</label>
            <div className="relative">
                <div className="absolute inset-y-0 start-0 flex items-center ps-3 pointer-events-none">
                    <svg className="w-4 h-4 text-gray-500 dark:text-gray-400" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 20">
                        <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="m19 19-4-4m0-7A7 7 0 1 1 1 8a7 7 0 0 1 14 0Z"/>
                    </svg>
                </div>
                <input value={question} onChange={(e) => {setQuestion(e.target.value)}} name="question" type="search" id="default-search" className="block w-full p-4 ps-10 text-sm text-gray-900 border border-gray-300 rounded-lg bg-gray-50 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500" placeholder="Ask a question" required />
                <button disabled={(progress !== 100)} type="submit" className="text-white absolute end-2.5 bottom-2.5 bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-sm px-4 py-2 dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800">Ask</button>
            </div>
        </form>
        <div className="w-full max-w-xl mx-auto">
          <span className="text-sm text-gray-900 dark:text-white"> Running on: {device} </span>
          <br />
          <span className="text-sm text-gray-900 dark:text-white"> Device memory: {memory} GB </span>
          <br />
          <span className="text-sm text-gray-900 dark:text-white"> Model Loading State: { isready ? `Done` : `${progress} %` } </span>
          <br />
          <span className="text-sm text-gray-900 dark:text-white"> Model: Llama-3.2-1B-Instruct is ready: { isready.toString() } </span>
          <br />
          {e !== "" ? <span className="text-sm text-red-500 dark:text-red-400"> {e} </span> : ""}
        </div>
        <div className="w-full max-w-xl mx-auto">
            <div className="flex flex-wrap gap-2">
                {/* {tokens.map((token, index) => (
                    <span key={index} className="px-2 py-1 text-sm text-gray-900 bg-gray-100 rounded-lg dark:text-white dark:bg-gray-800">{token}</span>
                ))} */}
                <span className="px-2 py-1 text-sm text-gray-900 bg-gray-100 rounded-lg dark:text-white dark:bg-gray-800">{token}</span>
            </div>
        </div>    
      </main>
      {/* <footer className="row-start-3 flex gap-6 flex-wrap items-center justify-center">
      </footer> */}
    </div>
  );
}
