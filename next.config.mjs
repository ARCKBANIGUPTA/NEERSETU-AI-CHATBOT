/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  // Vercel-specific optimizations
  experimental: {
    // Enable faster refresh for development
    fastRefresh: true,
  },
  // Enable SWC minification for better performance
  swcMinify: true,
  // Optimize for Vercel deployment
  output: 'standalone',
  // Enable compression
  compress: true,
  // Power by header for Vercel
  poweredByHeader: false,
}

export default nextConfig
