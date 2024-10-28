import { useEffect } from 'react'
import { useDispatch } from 'react-redux'
import { setPageTitle } from '../../features/common/headerSlice'
import RunPage from '../../features/run/index'

function InternalPage(){
    const dispatch = useDispatch()

    useEffect(() => {
        dispatch(setPageTitle({ title : "Run Optimization"}))
      }, [])


    return(
        <RunPage />
    )
}

export default InternalPage