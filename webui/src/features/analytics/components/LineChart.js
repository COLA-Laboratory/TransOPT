import React from 'react';
import { Line } from 'react-chartjs-2';
import TitleCard from '../../../components/Cards/TitleCard';

import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  Title,
  CategoryScale,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';

// 注册图表组件
ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Title, Tooltip, Legend, Filler);

const color = [
  "#2ec7c9", "#b6a2de", "#5ab1ef", "#ffb980", "#d87a80",
  "#8d98b3", "#e5cf0d", "#97b552", "#95706d", "#dc69aa",
  "#07a2a4", "#9a7fd1", "#588dd5", "#f5994e", "#c05050",
  "#59678c", "#c9ab00", "#7eb00a", "#6f5553", "#c14089"
];

const Trajectory = ({ TrajectoryData }) => {
  // 默认值处理，防止 undefined 错误
  const data = {
    datasets: TrajectoryData ? TrajectoryData.flatMap((item, index) => [
      // 线图数据集
      {
        label: `${item.name}`, // 数据集名称
        data: item.average.map(point => ({ x: point.FEs, y: point.y })), // 将数据点映射为 {x, y}
        borderColor: color[index % color.length], // 使用颜色数组中的颜色
        backgroundColor: color[index % color.length], // 线条颜色
        tension: 0.4, // 曲线平滑度
        fill: false, // 不填充区域
      },
      // 不确定性区域数据集
      {
        label: `${item.name} - uncertainty`, // 数据集名称
        data: item.uncertainty.map(point => ({ x: point.FEs, y: point.y })), // 不确定性区域数据
        borderColor: color[index % color.length], // 边框颜色
        backgroundColor: `${color[index % color.length]}33`, // 透明背景色表示不确定性
        tension: 0.4, // 曲线平滑度
        fill: true, // 填充区域
      },
    ]) : [],
  };

  const options = {
    scales: {
      x: {
        type: 'linear', // X轴类型为线性
        title: {
          display: true,
          text: 'FEs', // X轴标题
        },
      },
      y: {
        title: {
          display: true,
          text: 'Y', // Y轴标题
        },
        beginAtZero: true,
        nice: true,
        sync: true,
      },
    },
    plugins: {
      tooltip: {
        mode: 'index', // 工具提示框
        intersect: false,
        shared: true,
      },
      legend: {
        display: true,
        position: 'top',
        labels: {
          filter: (legendItem) => !legendItem.text.includes('uncertainty'), // 过滤掉包含 "uncertainty" 的标签
        },
      },
    },
  };

  return (
    <TitleCard title={"Convergence Trajectory"}>
      <Line data={data} options={options}/>
    </TitleCard>
    // <div style={{ width: '100%', height: '400px' }}>
    //   <Line data={data} options={options} />
    // </div>
  );
};

export default Trajectory;
