import { ChatOpenAI } from "@langchain/openai";

import { ChatPromptTemplate } from "@langchain/core/prompts";

import { Document } from "@langchain/core/documents";
import {createStuffDocumentsChain} from "langchain/chains/combine_documents";

import * as dotenv from 'dotenv';
dotenv.config();

const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.7,
});

const prompt = ChatPromptTemplate.fromTemplate(`
    Answer the user's Question. 
    Context: {context}
    Question: {input}
`);

//const chain = prompt.pipe(model);
const chain = await createStuffDocumentsChain({
    llm: model,
    prompt: prompt,

})

const documentA = new Document({
    pageContent: "LCEL is made by Akshit",
})

const documentB = new Document({
    pageContent: "Langchain lets go you are awsome and is created by openakshit",
})
const response  = await chain.invoke(
    {
        input : "What is LCEL, and what is langchain?",
        context: [documentA,documentB],
    }
);

console.log(response);