import { BaseMessageLike } from "@langchain/core/messages";
import { LangGraphRunnableConfig } from "@langchain/langgraph/web";
import { ToolCall } from "@langchain/core/messages/tool";
import type { ResponseComputerToolCall } from "openai/resources/responses/responses";
import { Page } from "playwright";
import { getHyperbrowserInstance, getToolOutputs } from "../utils.js";
import { CUAState, CUAUpdate } from "../types.js";

const sleep = (ms: number) =>
  new Promise((resolve) => {
    setTimeout(resolve, ms);
  });

export const CUA_KEY_TO_PLAYWRIGHT_KEY = {
  "/": "Divide",
  "\\": "Backslash",
  alt: "Alt",
  arrowdown: "ArrowDown",
  arrowleft: "ArrowLeft",
  arrowright: "ArrowRight",
  arrowup: "ArrowUp",
  backspace: "Backspace",
  capslock: "CapsLock",
  cmd: "Meta",
  ctrl: "Control",
  delete: "Delete",
  end: "End",
  enter: "Enter",
  esc: "Escape",
  home: "Home",
  insert: "Insert",
  option: "Alt",
  pagedown: "PageDown",
  pageup: "PageUp",
  shift: "Shift",
  space: " ",
  super: "Meta",
  tab: "Tab",
  win: "Meta",
};

const DUMMY_SCREENSHOT =
  "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKp//2Q==";

const translateKey = (key: string): string => {
  const lowerKey = key.toLowerCase();
  return lowerKey in CUA_KEY_TO_PLAYWRIGHT_KEY
    ? CUA_KEY_TO_PLAYWRIGHT_KEY[
        lowerKey as keyof typeof CUA_KEY_TO_PLAYWRIGHT_KEY
      ]
    : key;
};

async function handleFunctionToolCall(
  page: Page,
  functionToolCall: ToolCall
): Promise<BaseMessageLike> {
  const { name, args, id: call_id } = functionToolCall;

  try {
    if (name === "go_to_url") {
      await page.goto(args.url, {
        timeout: 15_000,
        waitUntil: "domcontentloaded",
      });
      await sleep(1_000);
      return {
        type: "tool",
        tool_call_id: call_id,
        content: `Successfully navigated to the URL: ${args.url}`,
        additional_kwargs: {
          type: "function_call_output",
          status: "completed",
        },
      };
    } else if (name === "get_current_url") {
      return {
        type: "tool",
        tool_call_id: call_id,
        content: page.url(),
        additional_kwargs: {
          type: "function_call_output",
          status: "completed",
        },
      };
    } else {
      throw new Error(`Unknown function tool call: ${name}`);
    }
  } catch (err) {
    console.error(`\n\nFailed to execute function call: ${name}\n\n`, err);
    console.error(
      `Function call details: ${JSON.stringify(functionToolCall, undefined, 2)}`
    );
    return {
      type: "tool",
      tool_call_id: call_id,
      content: `Failed to execute function call: ${name}`,
      additional_kwargs: { type: "function_call_output", status: "incomplete" },
    };
  }
}

async function handleComputerToolCall(
  page: Page,
  computerToolCall: ResponseComputerToolCall
): Promise<BaseMessageLike> {
  const { action, call_id } = computerToolCall;
  const actionType = action.type;

  try {
    switch (actionType) {
      case "click": {
        const { x, y, button } = action;
        switch (button) {
          case "back":
            await page.goBack({ timeout: 30_000 });
            break;
          case "forward":
            await page.goForward({ timeout: 30_000 });
            break;
          case "wheel":
            await page.mouse.wheel(x, y);
            break;
          case "left":
            await page.mouse.click(x, y, { button: "left" });
            break;
          case "right":
            await page.mouse.click(x, y, { button: "right" });
            break;
          default:
            throw new Error(`Unknown button: ${button}`);
        }
        break;
      }

      case "scroll": {
        const { x, y, scroll_x: scrollX, scroll_y: scrollY } = action;
        await page.mouse.move(x, y);
        await page.evaluate(`window.scrollBy(${scrollX}, ${scrollY})`);
        break;
      }

      case "keypress": {
        const { keys } = action;
        const mappedKeys = keys.map((key) => translateKey(key));
        for (const key of mappedKeys) {
          await page.keyboard.down(key);
        }
        for (const key of [...mappedKeys].reverse()) {
          await page.keyboard.up(key);
        }
        break;
      }

      case "type": {
        const { text } = action;
        // console.log(`Action: type text '${text}'`);
        await page.keyboard.type(text);
        break;
      }

      case "wait": {
        // console.log(`Action: wait`);
        await page.waitForTimeout(2000);
        break;
      }

      case "screenshot": {
        // Nothing to do as screenshot is taken at each turn
        // console.log(`Action: screenshot`);
        break;
      }

      case "double_click": {
        const { x, y } = action;
        // console.log(`Action: double click at (${x}, ${y})`);
        await page.mouse.click(x, y, { button: "left", clickCount: 2 });
        break;
      }

      case "drag": {
        const { path } = action;

        // console.log(`Action: drag with ${path.length} points`);

        if (path.length < 2) {
          throw new Error(
            "Invalid drag path: must contain at least a start and end point"
          );
        }

        await page.mouse.move(path[0].x, path[0].y);
        await page.mouse.down();

        for (const { x, y } of path) {
          await page.mouse.move(x, y);
          await page.waitForTimeout(40 + Math.floor(Math.random() * 40)); // Random delay between 40-79ms to simulate human dragging
        }

        await page.mouse.up();
        break;
      }

      case "move": {
        const { x, y } = action;
        // console.log(`Action: move to (${x}, ${y})`);
        await page.mouse.move(x, y);
        break;
      }

      default:
        throw new Error(`Unknown action type: ${actionType}`);
    }
    await sleep(1_000);
    const screenshot = await page.screenshot({ timeout: 15_000 });
    const b64Screenshot = Buffer.from(screenshot).toString("base64");
    const screenshotUrl = `data:image/png;base64,${b64Screenshot}`;
    return {
      type: "tool",
      tool_call_id: call_id,
      content: screenshotUrl,
      additional_kwargs: { type: "computer_call_output" },
    };
  } catch (error) {
    console.error(
      `\n\nFailed to execute computer call: ${actionType}\n\n`,
      error
    );
    console.error(`Computer call details: ${computerToolCall}`);
    return {
      type: "tool",
      tool_call_id: call_id,
      content: `data:image/jpeg;base64,${DUMMY_SCREENSHOT}`,
      additional_kwargs: { type: "computer_call_output", status: "incomplete" },
    };
  }
}

export async function takeHyperbrowserAction(
  state: CUAState,
  config: LangGraphRunnableConfig
): Promise<CUAUpdate> {
  if (!state.instanceId) {
    throw new Error("Can not take computer action without an instance ID.");
  }
  const message = state.messages[state.messages.length - 1];
  const toolOutputs = getToolOutputs(message) ?? [];

  // @ts-expect-error The message does include tool_calls or set to [] by default
  const functionCalls = message.tool_calls ?? [];

  if (toolOutputs.length === 0 && functionCalls.length === 0) {
    throw new Error(
      "Can not take computer action without a computer call or function call in the last message."
    );
  }

  const instance = await getHyperbrowserInstance(state.instanceId, config);
  if (instance.status !== "active") {
    throw new Error("Instance is not active.");
  }

  let { streamUrl, browserState } = state;

  if (!browserState) {
    throw new Error("Browser state not found.");
  }
  const { browser } = browserState;
  if (!browser) {
    throw new Error("Browser not found.");
  }
  const currentContext = browser.contexts()[0];
  let page = browserState.currentPage ?? currentContext.pages()[0];

  currentContext.on("page", (newPage: Page) => {
    page = newPage;
    if (!browserState) {
      browserState = {
        browser,
        currentPage: newPage,
      };
    } else {
      browserState.currentPage = newPage;
    }
  });

  if (!streamUrl) {
    streamUrl = instance.liveUrl;
    config.writer?.({
      streamUrl,
    });
  }

  let toolMsg: BaseMessageLike | undefined;
  const output = toolOutputs[toolOutputs.length - 1];

  for (const functionCall of functionCalls) {
    toolMsg = await handleFunctionToolCall(page, functionCall);
  }

  if (output) {
    if (output.type === "computer_call") {
      toolMsg = await handleComputerToolCall(page, output);
    } else {
      console.error("Unknown tool output type", output);
    }
  }

  return {
    messages: toolMsg ? [toolMsg] : [],
    instanceId: instance.id,
    streamUrl,
    browserState,
  };
}
