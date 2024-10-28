import { useEffect } from 'react'
import { useDispatch } from 'react-redux'
import ChatBot from '../../features/chatbot/ChatBot'
import { setPageTitle } from '../../features/common/headerSlice'

function InternalPage(){
    const dispatch = useDispatch()

    useEffect(() => {
        dispatch(setPageTitle({ title : "ChatOpt"}))
      }, [])


    return(
        <ChatBot/>
    )
}

export default InternalPage






