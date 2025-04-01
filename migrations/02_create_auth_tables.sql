-- Create authentication tables migration

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(36) PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(128) NOT NULL,
    -- password_salt VARCHAR(64) NOT NULL, -- REMOVED: Unnecessary with bcrypt
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    last_login TIMESTAMP NULL,
    is_active BOOLEAN DEFAULT TRUE
);

-- User roles table
CREATE TABLE IF NOT EXISTS user_roles (
    user_id VARCHAR(36) NOT NULL,
    role VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, role),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- User permissions table
CREATE TABLE IF NOT EXISTS user_permissions (
    user_id VARCHAR(36) NOT NULL,
    permission VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, permission),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- API keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    name VARCHAR(100) NOT NULL,
    key_hash VARCHAR(128) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP NULL,
    expires_at TIMESTAMP NULL,
    is_active BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create indexes for improved query performance
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_user_roles_user_id ON user_roles(user_id);
CREATE INDEX IF NOT EXISTS idx_user_permissions_user_id ON user_permissions(user_id);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);
CREATE INDEX IF NOT EXISTS idx_api_keys_created_at ON api_keys(created_at);
CREATE INDEX IF NOT EXISTS idx_api_keys_active_expires ON api_keys(is_active, expires_at);

-- Insert default admin user
-- Default password is 'admin' - THIS SHOULD BE CHANGED IMMEDIATELY IN PRODUCTION
-- REMOVED for security
-- INSERT INTO users (id, username, email, password_hash, password_salt)
-- VALUES (
--     'admin', -- Using a predictable UUID string for simplicity in example, generate properly
--     'admin',
--     'admin@example.com',
--     'c6002ebd59b5a6b5269755f4cc3b2f6deb5bd7e68a9a0b3e3248a1bf738a7887',  -- This is 'admin' with salt 'default_salt'
--     'default_salt'
-- ) ON DUPLICATE KEY UPDATE id = id; -- Using ON DUPLICATE KEY UPDATE might have unintended consequences, consider alternatives

-- Assign admin role to admin user
-- REMOVED for security
-- INSERT INTO user_roles (user_id, role)
-- VALUES ('admin', 'admin') ON DUPLICATE KEY UPDATE user_id = user_id; 