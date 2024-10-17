import { AutoModelForCausalLM, AutoTokenizer, pipeline, env, TextGenerationPipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers';
// import { TextGenerationPipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

console.log(navigator.gpu)

// // Create a feature-extraction pipeline
// const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
//   dtype: 'fp32',
//   device: 'webgpu', // <- Run on WebGPU
// });

// // Compute embeddings
// const texts = ['Hello world!', 'This is an example sentence.'];
// const embeddings = await extractor(texts, { pooling: 'mean', normalize: true });
// console.log(embeddings.tolist());
// // [
// //   [-0.020386919379234314, 0.025280799716711044, -0.0005662209587171674, ... ],
// //   [0.09812460094690323, 0.06781269609928131, 0.0625232458114624, ... ],
// // ]

// import { pipeline } from "@huggingface/transformers";

// Create a text generation pipeline

try {
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter) {
        console.log("WebGPU adapter available");
        const device = await adapter.requestDevice();
        console.log("WebGPU device available");
    } else {
        console.log("No suitable GPU adapter found.");
    }
} catch (error) {
    console.error("Error accessing WebGPU:", error.message || error);
}

try {
    env.localModelPath = './models';
    env.allowLocalModels = true;
    env.allowRemoteModels = false; 
    const tokenizer = await AutoTokenizer.from_pretrained( 
        "Llama-3.2-1B-Instruct",
        {
            dtype: 'q4f16',
            device: 'webgpu',
        }
    );
    // const model = await AutoModelForCausalLM.from_pretrained(
    //     "onnx-community/Llama-3.2-1B-Instruct", 
    //     {
    //         dtype: 'q4f16',
    //         device: 'webgpu',
    //     }
    // );

    // const generator = new TextGenerationPipeline({
    //     task: "text-generation",
    //     model: model,
    //     tokenizer: tokenizer
    // })

    const generator = await pipeline(
        "text-generation", 
        // model,
        // tokenizer: tokenizer,
        "Llama-3.2-1B-Instruct", 
        {
            dtype: 'q4f16',
            device: 'webgpu',
            // progress_callback: (beams) => {
            //     if (beams.status === "initiate"){
            //         console.log("Start initiate model");
            //     }else if (beams.status === "progress"){
            //         console.log(beams.progress);
            //     }else if (beams.status === "done"){
            //         console.log("Done");
            //     }
            // }
        }
    );

    console.log("pipeline created");
    // Define the list of messages
    // const messages = [
    //     { role: "system", content: "You are a helpful assistant." },
    //     { role: "user", content: "Tell me a joke." },
    // ];

    class TokenList {
        constructor() {
            this.data = []; // Using a Map to store key-value pairs
        }
    
        // `put` method to add data
        put(list) {
            this.data = list;
            console.log(tokenizer.decode(list[0], {skip_special_tokens: true,}))
        }
    
        get() {
            return this.data;
        }
    
        // `end` method to perform a final action, such as clearing the data
        end() {
            this.data = "";
        }
    }

    const messages = "tell me a joke";

    const tokenlist = new TokenList();

    const output = await generator(messages, { 
        max_new_tokens: 1024,
        streamer: tokenlist,
     })

    // console.log(tokenlist.get()[0])
    // console.log(tokenizer.decode(tokenlist.get()[0], {skip_special_tokens: true,}));

    console.log(output[0].generated_text);
    
} catch (error) {
    console.log("Error accessing WebGPU:", error);
}