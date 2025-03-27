import { test, expect } from "@jest/globals";
import { ChatOpenAI } from "@langchain/openai";
import { createCua } from "../index.js";
import { stopInstance } from "../utils.js";

test.skip("Can invoke the computer preview model", async () => {
  const model = new ChatOpenAI({
    model: "computer-use-preview",
    useResponsesApi: true,
  })
    .bindTools([
      {
        type: "computer-preview",
        display_width: 768,
        display_height: 1024,
        environment: "browser",
      },
    ])
    .bind({
      truncation: "auto",
    });

  const response = await model.invoke([
    {
      role: "system",
      content:
        "You're an advanced AI computer use assistant. The browser you are using is already initialized, and visiting google.com.",
    },
    {
      role: "user",
      content:
        "I'm looking for a new camera. Help me find the best one. It should be 4k resolution, by Cannon, and under $1000. I want a digital camera, and I'll be using it mainly for photography.",
    },
  ]);

  console.dir(response, { depth: Infinity });

  expect(response).toBeDefined();
});

test("It can use the agent to interact with the browser", async () => {
  let instanceId: string | undefined;
  const cuaGraph = createCua({ provider: "hyperbrowser", zdrEnabled: true, prompt: `You're an advanced AI computer use assistant. You are utilising a Chrome Browser with internet access.
              It is already open and running. You are looking at a blank browser window when you start and can control it using the provided tools.
              If you are on a blank page, you should use the go_to_url tool to navigate to the relevant website, 
              or if you need to search for something, go to https://www.google.com and search for it.` });
  try {
    const stream = await cuaGraph.stream(
      {
        messages: [
          {
            role: "system",
            content: `You're an advanced AI computer use assistant. You are utilising a Chrome Browser with internet access.
              It is already open and running. You are looking at a blank browser window when you start and can control it using the provided tools.
              If you are on a blank page, you should use the go_to_url tool to navigate to the relevant website, 
              or if you need to search for something, go to https://www.google.com and search for it.`,
          },
          {
            role: "user",
            content: `
              what is the price of nvidia stock?`,
          },
        ],
      },
      {
        streamMode: "updates",
      }
    );

    for await (const update of stream) {
      if (update.createVMInstance) {
        instanceId = update.createVMInstance.instanceId;
        console.log("----CREATE VM INSTANCE----\n", {
          VMInstance: {
            instanceId,
            streamUrl: update.createVMInstance.streamUrl,
          },
        });
      }

      if (update.takeComputerAction) {
        if (update.takeComputerAction?.messages?.[0]) {
          const message = update.takeComputerAction.messages[0];
          console.log("----TAKE COMPUTER ACTION----\n", {
            ToolMessage: {
              type: message.additional_kwargs?.type,
              tool_call_id: message.tool_call_id,
              content: `${message.content.slice(0, 50)}...`,
            },
          });
        }
      }

      if (update.callModel) {
        if (update.callModel?.messages) {
          const message = update.callModel.messages;
          const allOutputs = message.additional_kwargs?.tool_outputs;
          const toolCalls = message.tool_calls;
          if (allOutputs?.length) {
            const output = allOutputs[allOutputs.length - 1];
            console.log("----CALL MODEL----\n", {
              ComputerCall: {
                ...output.action,
                call_id: output.call_id,
              },
            });
            continue;
          } else if (toolCalls?.length) {
            const toolCall = toolCalls[toolCalls.length - 1];
            console.log("----CALL MODEL----\n", {
              FunctionCall: {
                ...toolCall,
              },
            });
          }
          console.log("----CALL MODEL----\n", {
            AIMessage: {
              content: message.content,
            },
          });
        }
      }
    }
  } finally {
    console.log("finished test", instanceId);
    /*if (instanceId) {
      console.log("Stopping instance with ID", instanceId);
      await stopInstance(instanceId);
    }*/
  }
});
