import React, { useState } from 'react';
import * as echarts from 'echarts';
import ReactECharts from 'echarts-for-react';
// import data from './data/Data.json';
import my_theme from './my_theme.json';

echarts.registerTheme('my_theme', my_theme.theme)

function Chart() {
  const option = {
    // option
  };
  
  return <ReactECharts
    option={option}
    style={{ height: 400 }}
    theme={"my_theme"}
  />;
};

export default Chart;