import {
  AIMessageChunk,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { LangGraphRunnableConfig } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import {
  CUAEnvironment,
  CUAState,
  CUAUpdate,
  getConfigurationWithDefaults,
} from "../types.js";
import {
  isComputerCallToolMessage,
  isFunctionCallToolMessage,
} from "../utils.js";

const _getOpenAIEnvFromStateEnv = (env: CUAEnvironment) => {
  switch (env) {
    case "web":
      return "browser";
    case "ubuntu":
      return "ubuntu";
    case "windows":
      return "windows";
    default:
      throw new Error(`Invalid environment: ${env}`);
  }
};

// Scrapybara does not allow for configuring this. Must use a hardcoded value.
const DEFAULT_DISPLAY_WIDTH = 1024;
const DEFAULT_DISPLAY_HEIGHT = 768;

const getAvailableTools = (config: LangGraphRunnableConfig) => {
  const { provider, environment, sessionParams } =
    getConfigurationWithDefaults(config);
  if (provider === "scrapybara") {
    return [
      {
        type: "computer_use_preview",
        display_width: DEFAULT_DISPLAY_WIDTH,
        display_height: DEFAULT_DISPLAY_HEIGHT,
        environment: _getOpenAIEnvFromStateEnv(environment),
      },
    ];
  } else if (provider === "hyperbrowser") {
    return [
      {
        type: "computer_use_preview",
        display_width: sessionParams?.screen?.width ?? DEFAULT_DISPLAY_WIDTH,
        display_height: sessionParams?.screen?.height ?? DEFAULT_DISPLAY_HEIGHT,
        environment: "browser",
      },
      {
        type: "function",
        function: {
          name: "go_to_url",
          description:
            "Navigate to a URL. Can be used when on a blank page to go to a specific URL or search engine.",
          parameters: {
            type: "object",
            properties: {
              url: {
                type: "string",
                description: "The fully qualified URL to navigate to",
              },
            },
            required: ["url"],
          },
        },
      },
      {
        type: "function",
        function: {
          name: "get_current_url",
          description: "Get the current URL",
          parameters: {
            type: "object",
            properties: {},
            required: [],
          },
        },
      },
    ];
  }
  throw new Error(`Unsupported provider: ${provider}`);
};

const _promptToSysMessage = (prompt: string | SystemMessage | undefined) => {
  if (typeof prompt === "string") {
    return { role: "system", content: prompt };
  }
  return prompt;
};

/**
 * Invokes the computer preview model with the given messages.
 *
 * @param {CUAState} state - The current state of the thread.
 * @param {LangGraphRunnableConfig} config - The configuration to use.
 * @returns {Promise<CUAUpdate>} - The updated state with the model's response.
 */
export async function callModel(
  state: CUAState,
  config: LangGraphRunnableConfig
): Promise<CUAUpdate> {
  const configuration = getConfigurationWithDefaults(config);

  const lastMessage = state.messages[state.messages.length - 1];
  let previousResponseId: string | undefined;
  const isLastMessageComputerCallOutput =
    isComputerCallToolMessage(lastMessage);
  const isLastMessageFunctionCallOutput =
    isFunctionCallToolMessage(lastMessage);

  if (
    (isLastMessageComputerCallOutput || isLastMessageFunctionCallOutput) &&
    !configuration.zdrEnabled
  ) {
    // Assume if the last message is a tool message, the second to last will be an AI message
    const secondToLast = state.messages[state.messages.length - 2];
    previousResponseId = secondToLast.response_metadata.id;
  }

  const model = new ChatOpenAI({
    model: "computer-use-preview",
    useResponsesApi: true,
  })
    .bindTools(getAvailableTools(config))
    .bind({
      truncation: "auto",
      previous_response_id: previousResponseId,
    });

  let response: AIMessageChunk;
  if (
    (isLastMessageComputerCallOutput || isLastMessageFunctionCallOutput) &&
    !configuration.zdrEnabled
  ) {
    if (isLastMessageFunctionCallOutput) {
      lastMessage.lc_kwargs = {
        tool_call_id: lastMessage.tool_call_id,
        content: lastMessage.content,
        type: "function_call_output",
      };
      lastMessage.id = undefined;
    }
    response = await model.invoke([lastMessage]);
  } else {
    const prompt = _promptToSysMessage(configuration.prompt);
    const messages = [...state.messages];
    for (let i = 0; i < messages.length; i += 1) {
      if (isFunctionCallToolMessage(messages[i])) {
        const castMsg = messages[i] as ToolMessage;
        messages[i].lc_kwargs = {
          tool_call_id: castMsg.tool_call_id,
          content: castMsg.content,
          type: "function_call_output",
        };
        messages[i].id = undefined;
      }
    }

    response = await model.invoke([...(prompt ? [prompt] : []), ...messages]);
  }

  return {
    messages: response,
  };
}
