#!/usr/bin/env node
"use strict";

const fs = require("fs");
const { spawnSync } = require("child_process");
const { engineBin, ensureEngine } = require("../scripts/install.js");

const args = process.argv.slice(2);

if (args[0] === "install") {
  process.exit(ensureEngine(args.slice(1)));
}

if (!fs.existsSync(engineBin())) {
  console.error("Lethyx is installed as an npm launcher, but its Python engine is not.");
  console.error("");
  console.error("Run:");
  console.error("  lethyx install");
  console.error("");
  console.error("Then configure it once with:");
  console.error("  lethyx setup");
  process.exit(1);
}

const result = spawnSync(engineBin(), args, { stdio: "inherit", env: process.env });
if (result.error) {
  console.error(result.error.message);
  process.exit(1);
}
process.exit(result.status == null ? 1 : result.status);
