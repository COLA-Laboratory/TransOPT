import React, { useState, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import cloneDeep from 'lodash.clonedeep';

function DynamicChart() {
  const DEFAULT_OPTION = {
    tooltip: {
      trigger: 'axis'
    },
    legend: {
      textStyle: {
        color: 'white'
      },
    },
    toolbox: {
      show: true,
      feature: {
        dataView: {readOnly: false},
        restore: {},
        saveAsImage: {}
      }
    },
    grid: {
      top: 60,
      left: 30,
      right: 60,
      bottom:30
    },
    dataZoom: {
      show: false,
      start: 0,
      end: 100
    },
    xAxis:{
      type: 'category',
      boundaryGap: true,
      data: (function (){
        let now = new Date();
        let res = [];
        let len = 50;
        while (len--) {
          res.unshift(now.toLocaleTimeString().replace(/^\D*/,''));
          now = new Date(now - 2000);
        }
        return res;
      })(),
      axisLine: {
          lineStyle: {
            color: 'white'
          }
        }
    },
    yAxis: {
      type: 'value',
      scale: true,
      name: '价格',
      max: 20,
      min: 0,
      boundaryGap: [0.2, 0.2],
      axisLine: {
        lineStyle: {
          color: 'white'
        }
      }
    },
    series: [
      {
        name:'最新成交价',
        type:'line',
        data:(function (){
          let res = [];
          let len = 0;
          while (len < 50) {
            res.push((Math.random()*10 + 5).toFixed(1) - 0);
            len++;
          }
          return res;
        })(),
        itemStyle: {
          color:'green',
          width: 10
        }
      }
    ]
  };

  const [option, setOption] = useState(DEFAULT_OPTION);

  function fetchNewData() {
    const axisData = (new Date()).toLocaleTimeString().replace(/^\D*/,'');
    const newOption = cloneDeep(option); // immutable
    const data = newOption.series[0].data;
    data.shift();
    data.push((Math.random() * 10 + 5).toFixed(1) - 0);

    newOption.xAxis.data.shift();
    newOption.xAxis.data.push(axisData);

    setOption(newOption);
  }

  useEffect(() => {
    const timer = setInterval(() => {
      fetchNewData();
    }, 1000);

    return () => clearInterval(timer);
  });

  return <ReactECharts
    option={option}
    style={{ height: 400 }}
  />;
};

export default DynamicChart;