import React from 'react';
import * as echarts from 'echarts';
import ReactECharts from 'echarts-for-react';
import BoxData from './data/BoxData.json';
import my_theme from './my_theme.json';

echarts.registerTheme('my_theme', my_theme.theme);

function Box({ BoxData }) {
  // Extract labels and data
  const dataLabel = Object.keys(BoxData);
  const data = Object.values(BoxData);

  // Configure the ECharts option
  const option = {
    dataset: [
      {
        source: data,
      },
      {
        transform: {
          type: 'boxplot',
          config: {
            itemNameFormatter: function (value) {
              return dataLabel[value.value];
            },
          },
        },
      },
      {
        fromDatasetIndex: 1,
        fromTransformResult: 1,
      },
    ],
    tooltip: {
      trigger: 'item',
      axisPointer: {
        type: 'shadow',
      },
    },
    toolbox: {
      feature: {
        saveAsImage: {},
      },
    },
    grid: {
      left: '10%',
      right: '10%',
      bottom: '15%',
    },
    xAxis: {
      type: 'category',
      // boundaryGap: true,
      nameGap: 30,
      axisLabel: {
        color: '#ffffff',
      },
      lineStyle: {
        color: 'black',
      },
    },
    yAxis: {
      type: 'value',
      name: 'value',
      lineStyle: {
        color: 'black',
      },
      axisLabel: {
        color: '#ffffff',
      },
      min: 'dataMin', // Set min to auto-scale
      max: 'dataMax', // Set max to auto-scale
    },
    series: [
      {
        name: 'boxplot',
        type: 'boxplot',
        datasetIndex: 1,
        itemStyle: {
          color: '#2EC7C9',
        },
      },
      {
        name: 'outlier',
        type: 'scatter',
        datasetIndex: 2,
        symbol: 'circle',
      },
    ],
  };

  return (
    <ReactECharts
      option={option}
      style={{ height: 500 }}
      theme="my_theme"
    />
  );
}

export default Box;
