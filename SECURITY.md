# Security Policy

## Supported versions

Security fixes are released for the latest version only. Update through Blender's extension
repository, or download the newest release from the
[releases page](https://github.com/dev-swarup/Blender-AI/releases).

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[security advisories](https://github.com/dev-swarup/Blender-AI/security/advisories/new) rather than in
a public issue, discussion or pull request.

Include:

- the Blender version, the Blender AI version and your operating system;
- the AI provider and URL type involved, if relevant (OpenAI-compatible or Anthropic);
- steps to reproduce, and the impact you expect.

Do not include real API keys or private scenes. If a proof of concept needs a key, use a throwaway
one. Once a fix is released the advisory is published, with credit to the reporter unless you prefer
to stay anonymous.

## Scope

In scope:

- Leaking an API key or other secret outside the Blender preferences.
- Sending data to a destination other than the URLs and servers the user configured.
- Bypassing *Ask before running Python*, so that a script runs without approval.
- Running code or writing files from a prompt, a `.blend` file, a model file or a tool result without the
  user's involvement beyond what this document describes.
- Flaws in the build, release or update pipeline that could ship a modified package.

Out of scope:

- `run_python` executing the Python it was asked to run. That is its purpose, and the approval setting
  exists for users who want to review each script.
- Behavior of third-party AI providers, MCP servers or helper add-ons.
- Vulnerabilities in Blender itself or in the bundled libraries, which belong with their own projects.
  Report an issue that Blender AI makes exploitable here.

## Security model

### Data sent from your machine

When you send a prompt, Blender AI sends the following to the AI provider you configured:

- your prompt and any images you attached;
- a short text summary of the scene (object names, types, transforms, materials, modifiers);
- tool results;
- screenshots of the 3D Viewport, only when *Let the AI see the viewport* is on.

Texture generation sends the texture prompt, and a reference image if you give one, to the image API.
Fetching the model list calls the provider's `/models` endpoint, and `import_from_url` downloads the model
files the AI asks for. Nothing is sent while Blender's *Allow Online Access* is off. The OpenAI Agents
SDK's tracing is disabled, so runs are not uploaded to OpenAI's dashboard.

A custom API URL receives everything listed above, including your key. Use only providers and gateways
you trust.

### API keys

Keys are stored in plain text in your Blender user preferences (`userpref.blend`), like other add-on
settings, and are never written into `.blend` files. Keys for MCP servers entered in the settings are
stored the same way. To keep keys out of the preferences, leave the fields empty and set
`OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in the environment.

### Running Python

`run_python` executes Python written by the AI inside Blender with full access, like Blender's Python
console. It can read and write files and start programs with your user's permissions, which is what lets
the AI do anything the purpose-built tools do not. If you work with untrusted `.blend` files, prompts or
MCP servers, turn on *Ask before running Python* and read each script before you allow it.

### MCP servers

MCP servers are programs (command servers, started on your computer) or online services (URL servers)
that you add yourself. They receive the arguments the AI sends to their tools. Stdio servers start with a
minimal environment plus the variables you configure for them. Add only servers you trust, and prefer the
configurations their publishers provide.

### Helper add-ons

Helper add-ons are installed only when you choose to, through Blender's extension system from
extensions.blender.org. Blender AI does not install anything else on its own.

### Permissions

The extension declares the permissions Blender shows at install time: network (the AI provider, model
lists, image generation and MCP), files (saving textures and downloaded models, starting MCP servers) and
clipboard (pasting into the prompt bar and MCP configurations).

### Releases

Releases are built by GitHub Actions from a tagged commit, and the package and `index.json` are attached
to the release. Install Blender AI only from this repository's releases or its extension repository
listing.

## Recommendations

- Use a key restricted to the models and spend you need, and revoke it if it leaks.
- Keep *Ask before running Python* on when you open files or prompts from people you do not know.
- Review MCP server commands before enabling them.
- Turn off *Allow Online Access* in Blender when you want no network activity at all.
