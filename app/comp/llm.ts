"use client";
import {
    SimpleChatModel,
    type BaseChatModelParams,
  } from "@langchain/core/language_models/chat_models";
  import { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
  import { AIMessageChunk, type BaseMessage } from "@langchain/core/messages";
  import { ChatGenerationChunk } from "@langchain/core/outputs";

  import { pipeline, env, TextGenerationPipeline, AutoTokenizer} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers/dist/transformers.min.js";

  class addToken {

    setTokens: (tokens: string) => void;
    setToken: (token: string) => void;
    tokenizer: AutoTokenizer;
    isEnd: boolean = false;
    ignore_prefix: boolean = true;
    start: boolean = false;

    constructor({setTokens, setToken, tokenizer, ignore_prefix}: {setTokens: (tokens: string) => void; setToken: (token: string) => void; tokenizer: AutoTokenizer; ignore_prefix: boolean}) {
        this.setTokens = setTokens;
        this.setToken = setToken;
        this.tokenizer = tokenizer;
        this.ignore_prefix = ignore_prefix;
        this.isEnd = false;
        this.start = false;
    }
    
    put(list: string[]) {
        if (this.ignore_prefix && !this.start) {
          this.start = true;
          return;
        }
        this.isEnd = false;
        this.setTokens((prevToken: string) => prevToken + this.tokenizer.decode(list[0], { skip_special_tokens: true }));
        this.setToken(this.tokenizer.decode(list[0], { skip_special_tokens: true }));
    }
    
    end() {
        this.isEnd = true;
    }

    checkEnd() {
        return this.isEnd;
    }
  }

  export interface LLMInput extends BaseChatModelParams {
    tokens: string;
    setToken: (token: string) => void;
    token: string;
    setTokens: (tokens: string) => void;
    n: number;
    model: string;
    pipeConfig: any;
  }
  
  export class LLM extends SimpleChatModel {
    n: number;
    model: string;
    pipeConfig: any;
    pipe: TextGenerationPipeline;
    tokens: string;
    setTokens: (tokens: string) => void;
    token: string;
    setToken: (token: string) => void;
    addToken!: addToken;
  
    constructor(fields: LLMInput) {
        super(fields);
        this.n = fields.n;
        this.model = fields.model ?? "Llama-3.2-1B-Instruct";
        this.pipeConfig = fields.pipeConfig;
        this.tokens = fields.tokens;
        this.setTokens = fields.setTokens;
        this.token = fields.token;
        this.setToken = fields.setToken;
    }
  
    _llmType() {
      return "LLM";
    }

    async init() {
        // env.localModelPath = './models';
        // env.allowLocalModels = true;
        // env.allowRemoteModels = true; 
        this.pipe = await pipeline("text-generation", this.model, this.pipeConfig);
        this.addToken = new addToken({ setTokens: this.setTokens, setToken: this.setToken, tokenizer: this.pipe.tokenizer, ignore_prefix: true });
    }

    reset(){
      this.addToken.start = false;
    }
  
    async _call(messages: BaseMessage[], options: this["ParsedCallOptions"], runManager?: CallbackManagerForLLMRun): Promise<string> {
      if (!messages.length) {
        throw new Error("No messages provided.");
      }
      if (typeof messages[0].content !== "string") {
        throw new Error("Multimodal messages are not supported.");
      }
      const input_length = messages[0].content.length;
      const output = await this.pipe(messages[0].content, {max_new_tokens: 1024, streamer: this.addToken})
      return output[0].generated_text.substring(input_length);
    }
  
    async *_streamResponseChunks(messages: BaseMessage[], options: this["ParsedCallOptions"], runManager?: CallbackManagerForLLMRun): AsyncGenerator<ChatGenerationChunk> {
      if (!messages.length) {
        throw new Error("No messages provided.");
      }
      if (typeof messages[0].content !== "string") {
        throw new Error("Multimodal messages are not supported.");
      }
      const output = await this.pipe(messages[0].content, {max_new_tokens: 1024, streamer: this.addToken})
      
      while (!this.addToken.checkEnd()) {
        yield new ChatGenerationChunk({
            message: new AIMessageChunk({
              content: this.token,
            }),
            text: this.token,
          });
        await runManager?.handleLLMNewToken(this.token);
      }
    }
  }
