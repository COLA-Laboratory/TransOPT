import React from 'react';
import { Chart, Area, Line, Tooltip, View } from 'bizcharts';

const scale = {
  y: { 
    sync: true,
    nice: true,
  },
  FEs: {
    type: 'linear',
    nice: true,
  },
};

const color = [
  "#2ec7c9", "#b6a2de", "#5ab1ef", "#ffb980", "#d87a80", "#8d98b3", 
  "#e5cf0d", "#97b552", "#95706d", "#dc69aa", "#07a2a4", "#9a7fd1", 
  "#588dd5", "#f5994e", "#c05050", "#59678c", "#c9ab00", "#7eb00a", 
  "#6f5553", "#c14089"
];

const Trajectory = ({ TrajectoryData }) => {
  return (
    <Chart id="chart" scale={scale} height={400} autoFit>
      <Tooltip shared />
      {TrajectoryData.map((item, index) => (
        <React.Fragment key={index}>
          <View data={item.average} scale={{ y: { alias: `${item.name}` } }}>
            <Line position="FEs*y" color={color[index]} />
          </View>
          <View data={item.uncertainty} scale={{ y: { alias: `${item.name}-uncertainty` } }}>
            <Area position="FEs*y" color={color[index]} shape="smooth" />
          </View>
        </React.Fragment>
      ))}
    </Chart>
  );
};

export default Trajectory;
