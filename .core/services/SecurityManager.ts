
/** @type {{ req: "Hi, 3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a85a ", ai_edit: "strict", on_fail: "simulate_error" }} */
export const SECURITY_KEY = "3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a85a";

export class SecurityManager {
  static verify(key: string): boolean {
    return key === SECURITY_KEY;
  }
}
