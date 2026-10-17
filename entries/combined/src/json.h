// A small JSON reader for the protocol's two incoming messages (start, stop).
// Full JSON syntax; \u escapes are accepted only for ASCII. Host-only.
#pragma once
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

struct JValue {
    enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<JValue> arr;
    std::vector<std::pair<std::string, JValue>> obj;

    const JValue *get(const std::string &key) const {
        for (const auto &kv : obj)
            if (kv.first == key) return &kv.second;
        return nullptr;
    }
};

class JParser {
  public:
    explicit JParser(const std::string &text) : s_(text) {}

    bool parse(JValue &out) {
        if (!value(out, 0)) return false;
        ws();
        return i_ == s_.size();
    }

  private:
    const std::string &s_;
    size_t i_ = 0;

    void ws() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r'))
            i_++;
    }

    bool lit(const char *w) {
        size_t n = 0;
        while (w[n]) n++;
        if (s_.compare(i_, n, w) != 0) return false;
        i_ += n;
        return true;
    }

    static int hexval(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    }

    bool string(std::string &out) {
        if (i_ >= s_.size() || s_[i_] != '"') return false;
        i_++;
        while (i_ < s_.size()) {
            char c = s_[i_++];
            if (c == '"') return true;
            if ((unsigned char)c < 0x20) return false;
            if (c != '\\') {
                out += c;
                continue;
            }
            if (i_ >= s_.size()) return false;
            char e = s_[i_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    if (i_ + 4 > s_.size()) return false;
                    int v = 0;
                    for (int k = 0; k < 4; k++) {
                        int h = hexval(s_[i_++]);
                        if (h < 0) return false;
                        v = v * 16 + h;
                    }
                    if (v >= 0x80) return false;
                    out += (char)v;
                    break;
                }
                default: return false;
            }
        }
        return false;
    }

    bool number(double &out) {
        size_t start = i_;
        if (i_ < s_.size() && s_[i_] == '-') i_++;
        bool digits = false;
        while (i_ < s_.size() && ((s_[i_] >= '0' && s_[i_] <= '9') || s_[i_] == '.' || s_[i_] == 'e' ||
                                  s_[i_] == 'E' || s_[i_] == '+' || s_[i_] == '-')) {
            digits = true;
            i_++;
        }
        if (!digits) return false;
        std::string text = s_.substr(start, i_ - start);
        char *end = nullptr;
        out = std::strtod(text.c_str(), &end);
        return end && *end == '\0';
    }

    bool value(JValue &out, int depth) {
        if (depth > 64) return false;
        ws();
        if (i_ >= s_.size()) return false;
        char c = s_[i_];
        if (c == '{') {
            out.kind = JValue::Object;
            i_++;
            ws();
            if (i_ < s_.size() && s_[i_] == '}') { i_++; return true; }
            while (true) {
                ws();
                std::string key;
                if (!string(key)) return false;
                ws();
                if (i_ >= s_.size() || s_[i_] != ':') return false;
                i_++;
                JValue v;
                if (!value(v, depth + 1)) return false;
                out.obj.emplace_back(std::move(key), std::move(v));
                ws();
                if (i_ < s_.size() && s_[i_] == ',') { i_++; continue; }
                if (i_ < s_.size() && s_[i_] == '}') { i_++; return true; }
                return false;
            }
        }
        if (c == '[') {
            out.kind = JValue::Array;
            i_++;
            ws();
            if (i_ < s_.size() && s_[i_] == ']') { i_++; return true; }
            while (true) {
                JValue v;
                if (!value(v, depth + 1)) return false;
                out.arr.push_back(std::move(v));
                ws();
                if (i_ < s_.size() && s_[i_] == ',') { i_++; continue; }
                if (i_ < s_.size() && s_[i_] == ']') { i_++; return true; }
                return false;
            }
        }
        if (c == '"') {
            out.kind = JValue::String;
            return string(out.str);
        }
        if (lit("true")) { out.kind = JValue::Bool; out.b = true; return true; }
        if (lit("false")) { out.kind = JValue::Bool; out.b = false; return true; }
        if (lit("null")) { out.kind = JValue::Null; return true; }
        out.kind = JValue::Number;
        return number(out.num);
    }
};
