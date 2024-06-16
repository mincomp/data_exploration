import os

import nbformat
from jupyter_client import KernelManager
from nbformat.v4 import new_code_cell, new_notebook, new_output
from openai import OpenAI

API_KEY = os.environ.get("OPENAI_API_KEY")
FILE = os.environ.get("FILE", "bird_migration.csv")
MODEL = os.environ.get("MODEL", "gpt-3.5-turbo")

client = OpenAI(api_key=API_KEY)
history = [
    {
        "role": "system",
        "content": f"You are a data scientist and you have been tasked with exploring a dataset represented by CSV, using Jupyter Notebook.",
    }
]
executed_cells = []

MAX_STEPS = 10


def send_message(message):
    history.append(message)
    output = client.chat.completions.create(
        model=MODEL,
        messages=history,
    )
    output_dict = {
        "role": output.choices[0].message.role,
        "content": output.choices[0].message.content,
    }
    history.append(output_dict)
    return output.choices[0].message.content


def get_plan(filename):
    return send_message(
        {
            "role": "user",
            "content": f"I have a dataset represented by CSV to explore. CSV file name is {filename}. What is your plan? Give me the list of steps only and no other information.",
        }
    )


def get_step():
    return send_message(
        {
            "role": "user",
            "content": f"Given the history of our conversation, what is the next step?",
        }
    )


def get_code():
    return send_message(
        {
            "role": "user",
            "content": f"Given the step you just decided, give me the piece of Python code to run. Raw code only, on other information. Remove formatting like python``` and ```.",
        }
    )


def get_commented_code(code):
    lines = code.split("\n")
    commented_lines = [
        f"# {line}" if not line.startswith("#") else line for line in lines
    ]
    return "\n".join(commented_lines)


class ExecutionResponse:
    def __init__(self, output, error):
        self.output = output
        self.error = error

    def __str__(self):
        return f"Output: {self.output}, Error: {self.error}"

    def __repr__(self):
        return self.__str__()


def run_to_complete(kc, code) -> ExecutionResponse:
    print(f"Code to run: {code}")
    msg_id = kc.execute(code)
    output = ""
    error = None
    cell = {
        "code": code,
    }
    cell_output = None

    # Wait for the result and display it
    while True:
        try:
            print("Waiting for message...")
            msg = kc.get_iopub_msg(timeout=1)
            content = msg["content"]
            msg_type = msg["header"]["msg_type"]

            # When a message with the text stream comes and it's the result of our execution
            if (
                msg_type == "execute_input"
            ):  # this message contains the code being executed
                print("Execution started.")
            elif (
                msg_type == "execute_result"
            ):  # this message contains the result of the execution
                print("Execution result:")
                for k, v in content["data"].items():
                    if k == "text/plain":
                        output += v + os.linesep
                        cell_output = v
                    else:
                        output += f"None text output: ({k})" + os.linesep
            elif msg_type == "display_data":
                print("Display data:")
                for k, v in content["data"].items():
                    if k == "text/plain":
                        output += v + os.linesep
                    else:
                        output += f"None text output: ({k})" + os.linesep
            elif msg_type == "status":
                print(f"Execution status {content}")
                if content["execution_state"] == "idle":
                    print("Execution finished.")
                    break
            elif msg_type == "stream":
                output += content["text"] + os.linesep
            elif msg_type == "error":
                print("An error occurred.")
                error = content
            else:
                print(f"Unknown message type. message: {msg}")
        except KeyboardInterrupt:
            print("Interrupted by user.")
            break
        except Exception as e:
            # If no messages are available, we'll end up here, but we can just continue and try again.
            print(f"Error: {e}")
            pass
    cell["output"] = cell_output
    cell["error"] = "\n".join(error["traceback"]) if error else None
    executed_cells.append(cell)
    return ExecutionResponse(output, error)


def main():
    plan = get_plan(FILE)
    executed_cells.append(
        {
            "code": get_commented_code(f"# Plan to explore dataset:\n{plan}"),
            "output": None,
            "error": None,
        }
    )
    print(f"Plan to explore dataset: {plan}")

    print("Starting kernel...")
    km = KernelManager(kernel_name="python3")
    km.start_kernel()

    print("Creating client...")
    # Create a client to interact with the kernel
    kc = km.client()
    kc.start_channels()

    # Ensure the client is connected before executing code
    kc.wait_for_ready()

    try:
        for _ in range(MAX_STEPS):
            step = get_step()
            executed_cells.append(
                {
                    "code": get_commented_code(f"# Next step: {step}"),
                    "output": None,
                    "error": None,
                }
            )
            print(f"Next step: {step}")

            code = get_code()

            result = run_to_complete(kc, code)
            print(result)
            history.append(
                {
                    "role": "user",
                    "content": f"This is the execution result: Output: {result.output}. Error: {result.error}",
                }
            )
    finally:
        # Cleanup
        print("Cleaning up...")
        km.shutdown_kernel()
        kc.stop_channels()

    print("Creating notebook...")
    nb = new_notebook()
    for cell in executed_cells:
        code_cell = new_code_cell(cell["code"])
        if cell["output"]:
            code_cell.outputs.append(new_output("stream", text=cell["output"]))
        if cell["error"]:
            code_cell.outputs.append(
                new_output("error", traceback=cell["error"].split("\n"))
            )
        nb.cells.append(code_cell)
    with open("data_exploration.ipynb", "w") as f:
        nbformat.write(nb, f)


if __name__ == "__main__":
    main()
