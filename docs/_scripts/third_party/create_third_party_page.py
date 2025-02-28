#!/usr/bin/env python
"""Create the third party page for the documentation."""

import argparse
from typing import List
from typing import TypedDict

import yaml

MARKDOWN = """\
[//]: # (This file is automatically generated using a script in docs/_scripts. Do not edit this file directly!)
# 🚀 Prebuilt Agents

LangGraph includes a prebuilt React agent. For more information on how to use it, 
check out our [how-to guides](https://langchain-ai.github.io/langgraphjs/how-tos/#prebuilt-react-agent).

If you’re looking for other prebuilt libraries, explore the community-built options 
below. These libraries can extend LangGraph's functionality in various ways.

## 📚 Available Libraries

[//]: # (This file is automatically generated using a script in docs/_scripts. Do not edit this file directly!)
{library_list}

## ✨ Contributing Your Library

Have you built an awesome open-source library using LangGraph? We'd love to feature 
your project on the official LangGraph documentation pages! 🏆

To share your project, simply open a Pull Request adding an entry for your package in our [packages.yml]({langgraph_url}) file.

**Guidelines**

- Your repo must be distributed as an installable package (e.g., PyPI for Python, npm 
  for JavaScript/TypeScript, etc.) 📦
- The repo should either use the Graph API (exposing a `StateGraph` instance) or 
  the Functional API (exposing an `entrypoint`).
- The package must include documentation (e.g., a `README.md` or docs site) 
  explaining how to use it.

We'll review your contribution and merge it in!

Thanks for contributing! 🚀
"""


class ResolvedPackage(TypedDict):
    name: str
    """The name of the package."""
    repo: str
    """Repository ID within github. Format is: [orgname]/[repo_name]."""
    monorepo_path: str | None
    """Optional: The path to the package root within a monorepo. Must be relative to the root of the monorepo."""
    weekly_downloads: int | None
    """The weekly download count of the package."""
    description: str
    """A brief description of what the package does."""


def generate_markdown(
    resolved_packages: List[ResolvedPackage], language: str,
  ) -> str:
    """Generate the markdown content for the third party page.

    Args:
        resolved_packages: A list of resolved package information.
        language: str
        has_weekly_downloads: bool

    Returns:
        The markdown content as a string.
    """
    # Update the URL to the actual file once the initial version is merged
    if language == "python":
        langgraph_url = (
            "https://github.com/langchain-ai/langgraph/blob/main/docs"
            "/_scripts/third_party_page/packages.yml"
        )
    elif language == "js":
        langgraph_url = (
            "https://github.com/langchain-ai/langgraphjs/blob/main/docs"
            "/_scripts/third_party/packages.yml"
        )
    else:
        raise ValueError(f"Invalid language '{language}'. Expected 'python' or 'js'.")

    sorted_packages = sorted(
        resolved_packages, key=lambda p: p["weekly_downloads"] or 0, reverse=True
    )

    rows = [
        "| Name | GitHub URL | Description | Weekly Downloads | Stars |",
        "| --- | --- | --- | --- | --- |",
    ]

    for package in sorted_packages:
        package_url = f"https://npmjs.com/package/{package['name']}"
        package_link = f"**[{package['name']}]({package_url})**"
        monorepo_path = package.get("monorepo_path", "")
        if monorepo_path:
            monorepo_path = monorepo_path[1:] if monorepo_path.startswith('/') else monorepo_path
            repo_url_suffix = f"/tree/main/{monorepo_path}"
        else:
            repo_url_suffix = ""
        repo_display_name = f"{package['repo']}{(' : ' + monorepo_path) if monorepo_path else ''}"
        repo_url = f"https://github.com/{package['repo']}{repo_url_suffix}"
        repo_link = f"[{repo_display_name}]({repo_url})"
        stars_badge = (
            f"https://img.shields.io/github/stars/{package['repo']}?style=social"
        )
        stars = f"![GitHub stars]({stars_badge})"

        downloads = package["weekly_downloads"] or "-"
        row = f"| {package_link} | {repo_link} | {package['description']} | {downloads} | {stars}"
        rows.append(row)
    markdown_content = MARKDOWN.format(
        library_list="\n".join(rows), langgraph_url=langgraph_url
    )
    return markdown_content


def main(input_file: str, output_file: str, language: str) -> None:
    """Main function to create the third party page.

    Args:
        input_file: Path to the input YAML file containing resolved package information.
        output_file: Path to the output file for the third party page.
        language: The language for which to generate the third party page.
    """
    # Parse the input YAML file
    with open(input_file, "r") as f:
        resolved_packages: List[ResolvedPackage] = yaml.safe_load(f)

    markdown_content = generate_markdown(resolved_packages, language)

    # Write the markdown content to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the third party page.")
    parser.add_argument(
        "input_file",
        help="Path to the input YAML file containing resolved package information.",
    )
    parser.add_argument(
        "output_file", help="Path to the output file for the third party page."
    )
    parser.add_argument(
        "--language",
        choices=["python", "js"],
        default="python",
        help="The language for which to generate the third party page. Defaults to 'python'.",
    )
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.language)
