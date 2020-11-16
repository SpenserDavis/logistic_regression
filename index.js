require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");
const LogisticRegression = require("./logistic-regression");
const plot = require("node-remote-plot");

const { features, labels } = loadCSV("./option_train.csv", {
  dataColumns: ["S", "K", "tau", "r"],
  labelColumns: ["black"],
  shuffle: true,
  converters: {
    black: (value) => {
      return value === "Over" ? 1 : 0;
    },
  },
});

const regression = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 30,
  batchSize: 100,
});

regression.train();

const testFeatures = loadCSV("./option_test.csv", {
  dataColumns: ["S", "K", "tau", "r"],
}).features;

regression.test(testFeatures);

plot({
  x: regression.costHistory.reverse(),
});
