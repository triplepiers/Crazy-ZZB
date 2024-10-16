# ZJUCTF 2024

笑鼠，今年和 SJTU 联办（阴暗的爬来玩耍）

啥也不会，只能阴暗的看看 Web 题（这对前端仔来说还是太难了）

## Intro - 容器题

1. 通过 `brew install websocat` 安装依赖

2. 使用 `websocat` 命令行程序直接连接交互或者监听

    - 安装后直接在命令行输入 `websocat -b wss://...` 即可进行交互

    - 输入 `websocat -b -E tcp-l:127.0.0.1:61234 wss://...` 即可在本地 61234 端口（可自行替换）开启 TCP 监听
  
        在另一个终端中通过 `nc 127.0.0.1 61234` 即可连接，pwntools 同理可用

## Web

### 1 easy Pentest (125Pt)

题面如下：

找一个趁手的工具，获取OSS里的flag吧(o^.^o)/

本题 flag 格式为 `flag{...}`

```text
AccessKey: LTAI5tLeHk1m8rLauyjs6Fyp
SecretKey: cfyDlZS7nobBOV0xGk0xaJ6IaOqm84
Target: https://oss-test-qazxsw.oss-cn-beijing.aliyuncs.com/fffffflllllaaaagggg.txt
```
---

1. 直接访问 [Target](https://oss-test-qazxsw.oss-cn-beijing.aliyuncs.com/fffffflllllaaaagggg.txt) 显然不对

    => 会给你蹦一个 AccessDenied，并且给出一个 [推荐参考文档](https://api.aliyun.com/troubleshoot?q=0003-00000005)

    不难认出这边用了阿里的 OSS 服务，下面开始查文档

2. 通过 API 访问，这边需要先安装依赖 `oss2`

    ```python
    import oss2

    auth = oss2.AuthV2(
        'LTAI5tLeHk1m8rLauyjs6Fyp',      # AccessKey
        'cfyDlZS7nobBOV0xGk0xaJ6IaOqm84' # SecretKey
    )

    # 这边的 Endpoint 和 Bucket Name 由 Target 拆解得到
    # 具体规则见：https://help.aliyun.com/zh/oss/user-guide/oss-domain-names
    bucket = oss2.Bucket(
        auth, 
        'https://oss-cn-beijing.aliyuncs.com', # Endpoint
        'oss-test-qazxsw'                      # Bucket Name
    )

    # 读取文件并打印
    flag = bucket.get_object('fffffflllllaaaagggg.txt')
    print(flag.read())

## Crypto

### 1 ShadOwTime

题面如下：

As a student who learns computer science, you should have known how to properly protect your programs.

```python
class Challenge():
    def __init__(self):
        self.secret = randrange(1, n)
        self.pubkey = Public_key(g, g * self.secret)
        self.privkey = Private_key(self.pubkey, self.secret)

    def sign_time(self):
        now: datetime = datetime.now()
        m: int = int(now.strftime("%m"))
        n: int = int(now.strftime("%S"))
        current: str = f"{m}:{n}"
        msg: str = f"Current time is {current}"
        hsh: bytes = sha1(msg.encode()).digest()
        sig: Signature = self.privkey.sign(bytes_to_long(hsh), randrange(1, n))
        return {"msg": msg, "r": hex(sig.r), "s": hex(sig.s)}

    def verify(self):
        msg: str = input("Enter the message: ")
        r: str = input("Enter r in hexadecimal form: ")
        s: str = input("Enter s in hexadecimal form: ")
        hsh: bytes = sha1(msg.encode()).digest()
        sig_r: int = int(r, 16)
        sig_s: int = int(s, 16)
        sig: Signature = Signature(sig_r, sig_s)
        return self.pubkey.verifies(bytes_to_long(hsh), sig)

    def challenge(self):
        print(banner)
        for _ in range(2):
            option: str = input("Enter your option: ")
            if option == 'sign_time':
                print(self.sign_time())
            elif option == 'verify':
                if self.verify():
                    print(f"The flag is: {flag}")
                    exit(0)
                else:
                    print("Invalid signature!")
            else:
                print("Invalid option!")
        exit(0)
```

虽然我完全没看懂这边的 code，但是

1. 选择 `sign_time`，然后会返回一个 JSON

2. 选择 `verify`，然后把 (1) 中返回的 `msg, r, s` 依次输回去

3. 没了