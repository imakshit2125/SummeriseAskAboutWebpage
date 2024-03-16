import express from "express";
import fetch from "node-fetch"; // Import fetch
import { ChatOpenAI } from "@langchain/openai";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrievalChain } from "langchain/chains/retrieval";
import * as dotenv from "dotenv";
dotenv.config();

// Instantiate Model
const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0.7,
});
const app = express();


app.get("/fetch-webpage", async (req, res) => {
  try {
    const url = req.query.url;
    console.log(url);
    const response = await fetch(url);
    
    const content = await response.text();
    const resp = "The Requested URL fetched is " + url;
    res.send(resp);
  } catch (error) {
    console.error("Error while fetching webpage:", error);
    res.status(500).send("Error while fetching webpage");
  }
});

app.get("/answer", async (req, res) => {
  try {
    const question = req.query.input;

    // Create prompt
    const prompt = ChatPromptTemplate.fromTemplate(`Answer the user's question from the following context: 
      {context}
      Question: ${question}`);

    // Create Chain
    const chain = await createStuffDocumentsChain({
      llm: model,
      prompt,
    });

    console.log(req.query.url);
    //Use Cheerio to scrape content from webpage and create documents
    const loader = new CheerioWebBaseLoader(req.query.url);
    const docs = await loader.load();

    // console.log(docs[0])

    // Text Splitter
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 200,
      chunkOverlap: 40,
    });
    const splitDocs = await splitter.splitDocuments(docs);

    // Instantiate Embeddings function
    const embeddings = new OpenAIEmbeddings();

    // Create Vector Store
    const vectorstore = await MemoryVectorStore.fromDocuments(
      splitDocs,
      embeddings
    );

    // Create a retriever from vector store
    const retriever = vectorstore.asRetriever({ k: 2 });

    // Create a retrieval chain
    const retrievalChain = await createRetrievalChain({
      combineDocsChain: chain,
      retriever,
    });

    // Retrieve the answer
    const response = await retrievalChain.invoke({
      input: question,
    });

    res.json({ answer: response.answer });
    console.log(response)
    res.send(response.answer);
  } catch (error) {
    console.error("Error while fetching answer:", error);
    res.status(500).send("Error while fetching answer");
  }
});

// Serve static files
app.use(express.static("./"));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port -> ${PORT}`);
});
