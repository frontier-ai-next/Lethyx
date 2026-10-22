"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");

function lethyxHome() {
  return process.env.LETHYX_HOME
    ? path.resolve(process.env.LETHYX_HOME)
    : path.join(os.homedir(), ".lethyx");
}

function venvDir() {
  return process.env.LETHYX_VENV
    ? path.resolve(process.env.LETHYX_VENV)
    : path.join(lethyxHome(), "python");
}

function venvPython() {
  return process.platform === "win32"
    ? path.join(venvDir(), "Scripts", "python.exe")
    : path.join(venvDir(), "bin", "python");
}

function engineBin() {
  return process.platform === "win32"
    ? path.join(venvDir(), "Scripts", "lethyx.exe")
    : path.join(venvDir(), "bin", "lethyx");
}

function mcpBin() {
  return process.platform === "win32"
    ? path.join(venvDir(), "Scripts", "lethyx-mcp.exe")
    : path.join(venvDir(), "bin", "lethyx-mcp");
}

function run(command, args, options) {
  const result = spawnSync(command, args, {
    stdio: "inherit",
    env: process.env,
    ...options,
  });
  if (result.error) {
    console.error(result.error.message);
    return 1;
  }
  return result.status == null ? 1 : result.status;
}

function findPython() {
  if (process.env.LETHYX_PYTHON) return [process.env.LETHYX_PYTHON];
  const candidates =
    process.platform === "win32"
      ? [["py", "-3.12"], ["py", "-3"], ["python"], ["python3"]]
      : [["python3.12"], ["python3"], ["python"]];
  for (const candidate of candidates) {
    const probe = spawnSync(candidate[0], [...candidate.slice(1), "-c", "import sys; print(sys.version_info[:2])"], {
      encoding: "utf8",
    });
    if (probe.status === 0 && /\(3, 1[2-9]\)/.test(probe.stdout || "")) {
      return candidate;
    }
  }
  return null;
}

function parseArgs(args) {
  const parsed = { from: "", semantic: null, force: false };
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--from" || arg === "--spec") parsed.from = args[++index] || "";
    else if (arg === "--semantic") parsed.semantic = true;
    else if (arg === "--no-semantic") parsed.semantic = false;
    else if (arg === "--force") parsed.force = true;
  }
  return parsed;
}

function defaultSpec(semantic) {
  return semantic ? "lethyx[mcp,semantic]" : "lethyx[mcp]";
}

function ensureEngine(rawArgs) {
  const options = parseArgs(rawArgs || []);
  if (fs.existsSync(engineBin()) && !options.force) {
    console.log(`Lethyx is already installed at ${engineBin()}`);
    if (options.semantic === true) {
      console.log("To add the optional embedding runtime, run:");
      console.log("  lethyx install --semantic --force");
    } else {
      console.log("Use 'lethyx install --force' to reinstall or change extras.");
    }
    return 0;
  }
  const python = findPython();
  if (!python) {
    console.error("Lethyx needs Python 3.12 or newer.");
    console.error("Install Python from https://www.python.org/downloads/ and run this again.");
    return 1;
  }
  fs.mkdirSync(lethyxHome(), { recursive: true });
  if (!fs.existsSync(venvPython())) {
    console.log(`Creating a Python environment in ${venvDir()}`);
    const created = run(python[0], [...python.slice(1), "-m", "venv", venvDir()]);
    if (created !== 0) return created;
  }
  let spec = options.from || process.env.LETHYX_PIP_SPEC || defaultSpec(options.semantic === true);
  if (options.semantic === true && options.from && !/\[/.test(spec)) {
    console.log("Note: --semantic is ignored with a custom --from spec; include the extra in the spec.");
  }
  console.log(`Installing ${spec}`);
  const installed = run(venvPython(), ["-m", "pip", "install", "--upgrade", spec]);
  if (installed !== 0) {
    console.error("\nThe Python package could not be installed.");
    console.error('For a source checkout use: lethyx install --from "C:\\\\path\\\\to\\\\Lethyx"');
    return installed;
  }
  const marker = {
    installedAt: new Date().toISOString(),
    spec,
    venv: venvDir(),
    python: venvPython(),
    lethyx: engineBin(),
    mcp: mcpBin(),
  };
  fs.writeFileSync(path.join(lethyxHome(), "npm.json"), JSON.stringify(marker, null, 2) + "\n");
  console.log(`\nLethyx installed. Engine: ${engineBin()}`);
  console.log("Next: lethyx setup");
  return 0;
}

module.exports = {
  ensureEngine,
  engineBin,
  findPython,
  lethyxHome,
  mcpBin,
  venvDir,
  venvPython,
};
