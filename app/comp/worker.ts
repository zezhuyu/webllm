import { pipeline, env, TextGenerationPipeline, PipelineType } from "@huggingface/transformers";

// Skip local model check
env.allowLocalModels = false;

// Use the Singleton pattern to enable lazy construction of the pipeline.
class PipelineSingleton {
    static task = 'text-generation' as PipelineType;
    static model = 'onnx-community/Llama-3.2-1B-Instruct';
    static instance: TextGenerationPipeline | null = null;

    static async getInstance(progress_callback: Function | undefined = undefined) {
        if (this.instance === null) {
            this.instance = await pipeline(this.task, this.model, {
                dtype: 'fp32',
                device: 'webgpu',
                progress_callback,
                useFast: true // Add this option to use the fast version of the pipeline

            });
        }
        return this.instance;
    }
}

// Listen for messages from the main thread
self.addEventListener('message', async (event) => {
    // Retrieve the classification pipeline. When called for the first time,
    // this will load the pipeline and save it for future use.
    let classifier = await PipelineSingleton.getInstance(x => {
        // We also add a progress callback to the pipeline so that we can
        // track model loading.
        self.postMessage(x);
    });

    // Actually perform the classification
    let output = await classifier(event.data.text);

    // Send the output back to the main thread
    self.postMessage({
        status: 'complete',
        output: output,
    });
});
