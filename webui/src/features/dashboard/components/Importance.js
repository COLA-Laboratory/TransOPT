import React, {useState, useEffect} from 'react';

function Importance() {  
  const [imageUrl, setImageUrl] = useState(require('../../../exp_pictures/parameter_network.png'));

  useEffect(() => {
    // 在组件加载时自动更换图片
    setImageUrl(require('../../../exp_pictures/parameter_network.png'));
  }, []);

  return <img src={imageUrl + '?' + new Date().getTime()} alt="network" style={{ width: 'auto', height: 'auto', maxWidth: '100%', maxHeight: '100%' }} /> 
};

export default Importance;