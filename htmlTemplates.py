css = '''
<style>
    
  .chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    animation: fadeIn 0.5s ease;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    transition: background-color 0.3s ease;
  }

  .chat-message.user {
    background-color: #92d1f0;
    transition: background-color 0.3s ease;
  }

  .chat-message.bot {
    background-color: #1d99f2;
    transition: background-color 0.3s ease;
  }

  .chat-message:hover {
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
  }

  .chat-message.user:hover {
    background-color: #384252;
  }

  .chat-message.bot:hover {
    background-color: #5a6880;
  }

  .chat-message .avatar {
    width: 20%;
  }

  .chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
  }

  .chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #fff;
  }

  @keyframes fadeIn {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }
'''


bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
    <img src="https://i.ibb.co/jV4FJL1/christopher-campbell-r-DEOVt-E7v-Os-unsplash.jpg" style="height: 78px;width: 78px; border-radius: 50%; object-fit: cover;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
