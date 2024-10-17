"use client";
import type {
    PretrainedOptions,
    FeatureExtractionPipelineOptions,
    FeatureExtractionPipeline,
  } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers/dist/transformers.min.js";
  import {pipeline, env} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers/dist/transformers.min.js";
  import { Embeddings, type EmbeddingsParams } from "@langchain/core/embeddings";
  import { chunkArray } from "@langchain/core/utils/chunk_array";
  
  export interface HuggingFaceTransformersEmbeddingsParams extends EmbeddingsParams {
    modelName: string;
    model: string;
    timeout?: number;
    batchSize?: number;
    stripNewLines?: boolean;
    pretrainedOptions?: PretrainedOptions;
    pipelineOptions?: FeatureExtractionPipelineOptions;
  }
  
  /**
   * @example
   * ```typescript
   * const model = new HuggingFaceTransformersEmbeddings({
   *   model: "Xenova/all-MiniLM-L6-v2",
   * });
   *
   * // Embed a single query
   * const res = await model.embedQuery(
   *   "What would be a good company name for a company that makes colorful socks?"
   * );
   * console.log({ res });
   *
   * // Embed multiple documents
   * const documentRes = await model.embedDocuments(["Hello world", "Bye bye"]);
   * console.log({ documentRes });
   * ```
   */
  export class HuggingFaceTransformersEmbeddings extends Embeddings implements HuggingFaceTransformersEmbeddingsParams{
    modelName = "Xenova/all-MiniLM-L6-v2";
  
    model = "Xenova/all-MiniLM-L6-v2";
  
    batchSize = 512;
  
    stripNewLines = true;
  
    timeout?: number;
  
    pretrainedOptions?: PretrainedOptions;
  
    pipelineOptions?: FeatureExtractionPipelineOptions;
  
    private pipelinePromise: Promise<FeatureExtractionPipeline> | undefined;

    private pipe: FeatureExtractionPipeline | undefined;
  
    constructor(fields?: Partial<HuggingFaceTransformersEmbeddingsParams>) {
      super(fields ?? {});
  
      this.modelName = fields?.model ?? fields?.modelName ?? this.model;
      this.model = this.modelName;
      this.stripNewLines = fields?.stripNewLines ?? this.stripNewLines;
      this.timeout = fields?.timeout;
      this.pretrainedOptions = fields?.pretrainedOptions ?? {};
      this.pipelineOptions = {pooling: "mean", normalize: true, ...fields?.pipelineOptions,};
    }
  
    async init() {
        env.localModelPath = './models';
        env.allowLocalModels = true;
        env.allowRemoteModels = true; 
        this.pipe = await pipeline("feature-extraction", this.model, this.pretrainedOptions);
    }

    async embedDocuments(texts: string[]): Promise<number[][]> {
      const batches = chunkArray(
        this.stripNewLines ? texts.map((t) => t.replace(/\n/g, " ")) : texts,
        this.batchSize
      );
  
      const batchRequests = batches.map((batch) => this.runEmbedding(batch));
      const batchResponses = await Promise.all(batchRequests);
      const embeddings: number[][] = [];
  
      for (let i = 0; i < batchResponses.length; i += 1) {
        const batchResponse = batchResponses[i];
        for (let j = 0; j < batchResponse.length; j += 1) {
          embeddings.push(batchResponse[j]);
        }
      }
  
      return embeddings;
    }
  
    async embedQuery(text: string): Promise<number[]> {
      const data = await this.runEmbedding([
        this.stripNewLines ? text.replace(/\n/g, " ") : text,
      ]);
      return data[0];
    }
  
    private async runEmbedding(texts: string[]) {
      return this.caller.call(async () => {
        const output = await this.pipe(texts, this.pipelineOptions);
        return output.tolist();
      });
    }
  }