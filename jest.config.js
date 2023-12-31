/**
 * @type {import("@jest/types").Config.ProjectConfig}
 */
module.exports = {
  testTimeout: 10 * 1000,
  transform: {
    ".(js|ts|tsx)": "ts-jest",
  },
  collectCoverageFrom: ["src/**/*.ts", "!**/node_modules/**"],
  coveragePathIgnorePatterns: ["node_modules/"],
  testEnvironment: "node",
  testRegex: "/test/.*\\.test\\.[tj]s$",
  moduleFileExtensions: ["ts", "js", "json"],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 85,
      lines: 85,
      statements: 85,
    },
  },
  reporters: ["default", ["jest-junit", { outputDirectory: "coverage" }]],
};
