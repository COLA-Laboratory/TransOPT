import { useEffect } from 'react'
import { useDispatch } from 'react-redux'
import { setPageTitle } from '../../features/common/headerSlice'
import Seldata from '../../features/seldata/index'

function InternalPage(){
    const dispatch = useDispatch()

    useEffect(() => {
        dispatch(setPageTitle({ title : "Select Datasets"}))
      }, [])


    return(
        <Seldata />
    )
}

export default InternalPage