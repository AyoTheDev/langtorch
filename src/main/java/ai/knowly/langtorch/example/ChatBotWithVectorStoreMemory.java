package ai.knowly.langtorch.example;

import ai.knowly.langtorch.capability.module.openai.VectorStoreChatCapability;
import ai.knowly.langtorch.store.memory.conversation.ConversationMemory;
import com.google.common.flogger.FluentLogger;

import java.io.IOException;

import static ai.knowly.langtorch.example.ExampleUtils.readInputUntilEXIT;

public class ChatBotWithVectorStoreMemory {
  private static final FluentLogger logger = FluentLogger.forEnclosingClass();

  //TODO: This will need to check for data ingestion first, check if DB is empty...
  //TODO: if it is empty we need to run a utility method that uploads a PDF document from a given folder

  public static void main(String[] args) throws IOException {
    // Reading the key from the environment variable under Resource folder(.env file, OPENAI_API_KEY
    // field)
    VectorStoreChatCapability chatBot =
            VectorStoreChatCapability.create()
                    .withMemory(ConversationMemory.builder().build())
                    .withVerboseMode();
    readInputUntilEXIT(logger, chatBot);
  }
}
