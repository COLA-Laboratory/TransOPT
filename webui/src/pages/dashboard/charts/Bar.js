import React from 'react';
import * as echarts from 'echarts';
import ReactECharts from 'echarts-for-react';
import my_theme from './my_theme.json';

echarts.registerTheme('my_theme', my_theme.theme)


function Bar({BarData}) {
//   console.log("BarData:", BarData)
  let seriesArr = [];
  let Xlabel = [];
  BarData.map((item, index) => {
    Xlabel.push(item.name)
    let obj = {};
    obj.name = item.name;
    obj.type = "bar";
    obj.barWidth = "50%";
    obj.barGap = "-100%";
    obj.data = [];
    for (var i = 0; i <= index; i++) {
        if (i !== index) {
            obj.data.push(0);
        } else {
            obj.data.push(item.value);
        }
    }
    seriesArr.push(obj);
  }
  )
  const option = {
    // option
    legend: {
        textStyle:{
            color: "#ffffff",
        }
    },
    tooltip:{
        axisPointer:{
            type: "line",
        }
    },
    toolbox:{
        feature: {
            saveAsImage: {}
        }
    },
    xAxis: {
        type: 'category',
        data: Xlabel,
        axisLabel: {
            color: '#ffffff'
        },
        lineStyle: {
            color: 'white'
        },
    },
    yAxis: {
        type: 'value',
        nameTextStyle:{
            color: "#ffffff",
        },
        axisLabel: {
          color: '#ffffff'
        },
        lineStyle: {
            color: 'white'
        },
    },
    series: seriesArr,
  };
  
  return <ReactECharts
    option={option}
    style={{ height: 400 }}
    theme={"my_theme"}
  />;
};

export default Bar;