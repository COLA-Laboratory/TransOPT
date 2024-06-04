import React from 'react';
import * as echarts from 'echarts';
import ReactECharts from 'echarts-for-react';
import my_theme from './my_theme.json';

echarts.registerTheme('my_theme', my_theme.theme)

function Scatter({ScatterData}) {
  const convertToseries = Object.keys(ScatterData).map(key=>({
    name: key,
    data: ScatterData[key],
    type: 'scatter',
    symbol: 'circle'
  }))  


  const option = {
    // option
    legend: {
        textStyle: {
            color: '#ffffff'
          }
    },
    toolbox:{
      feature: {
          saveAsImage: {}
      }
    },
    xAxis: {
        type:"value",
        show: false, 
        axisLabel: {
            color: '#ffffff'
        },
        lineStyle: {
            color: 'white'
        },
        splitLine: { show: false },
    },
    yAxis: {
        type: "value",
        show: false, 
        axisLabel: {
            color: '#ffffff'
        },
        lineStyle: {
            color: 'white'
        },
        splitLine: { show: false },
    },
    series: convertToseries
  };
  
  return <ReactECharts
    option={option}
    style={{ height: 400 }}
    theme={"my_theme"}
  />;
};

export default Scatter;