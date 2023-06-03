package ai.knowly.langtorch.capability.module.openai;

import ai.knowly.langtorch.capability.modality.text.ChatCompletionLLMCapability;
import ai.knowly.langtorch.preprocessing.parser.ChatMessageToStringParser;
import ai.knowly.langtorch.preprocessing.parser.Parser;
import ai.knowly.langtorch.preprocessing.parser.StringToMultiChatMessageParser;
import ai.knowly.langtorch.processor.module.openai.chat.OpenAIChatProcessor;
import ai.knowly.langtorch.schema.chat.ChatMessage;
import ai.knowly.langtorch.schema.text.MultiChatMessage;
import ai.knowly.langtorch.store.memory.conversation.ConversationMemory;
import ai.knowly.langtorch.store.vectordb.integration.pinecone.PineconeVectorStore;

import java.util.Optional;

/** A Vectorstore chat capability unit that leverages openai api to generate responses in context with documents stored on a Vector Database*/
public class VectorStoreChatCapability extends ChatCompletionLLMCapability<String, String> {

  //TODO: might not be the correct spot for this...?
  PineconeVectorStore vectorStore;
  private VectorStoreChatCapability(OpenAIChatProcessor openAIChatProcessor) {
    super(
        openAIChatProcessor,
        Optional.of(StringToMultiChatMessageParser.create()),
        Optional.of(ChatMessageToStringParser.create()));
  }

  private VectorStoreChatCapability() {
    super(
        OpenAIChatProcessor.create(),
        Optional.of(StringToMultiChatMessageParser.create()),
        Optional.of(ChatMessageToStringParser.create()));
  }

  public static VectorStoreChatCapability create() {
    return new VectorStoreChatCapability();
  }

  public static VectorStoreChatCapability create(OpenAIChatProcessor openAIChatProcessor) {
    return new VectorStoreChatCapability(openAIChatProcessor);
  }

  public static VectorStoreChatCapability create(String openAIKey) {
    return new VectorStoreChatCapability(OpenAIChatProcessor.create(openAIKey));
  }

  //TODO: Maybe a similar method here called withVectorStore
  //TODO: I guess that a dev shouldn't be able to create this without a vector store interface
  public VectorStoreChatCapability withMemory(ConversationMemory conversationMemory) {
    super.withMemory(conversationMemory);
    return this;
  }

  @Override
  public VectorStoreChatCapability withVerboseMode() {
    super.withVerboseMode();
    return this;
  }

  @Override
  public ChatCompletionLLMCapability<String, String> withInputParser(
      Parser<String, MultiChatMessage> inputParser) {
    super.withInputParser(inputParser);
    return this;
  }

  @Override
  public ChatCompletionLLMCapability<String, String> withOutputParser(
      Parser<ChatMessage, String> outputParser) {
    super.withOutputParser(outputParser);
    return this;
  }
}
