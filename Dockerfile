# Install Rust and Cargo
RUN apt-get update && apt-get install -y curl && \
    curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    export PATH="$HOME/.cargo/bin:$PATH"